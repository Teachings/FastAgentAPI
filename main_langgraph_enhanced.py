# main_langgraph_enhanced.py
import os
import json
import uuid
import time
import asyncio
import re
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from urllib.parse import urlparse, urlunparse, urljoin
import traceback
from typing import List, Dict, Any, Tuple, Optional, TypedDict, Literal
import jsonschema
from jsonschema import ValidationError

# --- LangGraph Imports ---
# Install with: pip install langgraph langchain_core
from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage # Used in helpers, but not directly in state

load_dotenv()

# --- Configuration ---
# Backend API URL (MUST include the base path like /v1)
YOUR_BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:5000/v1")
# Backend API Key (optional, depends on backend auth)
YOUR_BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", None)
# Port for this wrapper server (Docker typically exposes the port specified here)
WRAPPER_PORT = int(os.environ.get("PORT", 8000))
# Max retries for backend calls
DEFAULT_MAX_RETRIES = int(os.environ.get("DEFAULT_MAX_RETRIES", 1))

# --- Logging Control ---
# Use logging level based on VERBOSE_LOGGING
VERBOSE_LOGGING = os.environ.get("VERBOSE_LOGGING", "true").lower() == "true"
LOG_LEVEL = logging.DEBUG if VERBOSE_LOGGING else logging.INFO
# Get log file path from environment variable, default to 'app.log' in the current dir (or /app/logs/app.log in Docker)
DEFAULT_LOG_PATH = "/app/logs/app.log" # Default path inside the container
LOG_FILE_PATH = os.environ.get("LOG_FILE_PATH", DEFAULT_LOG_PATH)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Get root logger or a specific name
logger.setLevel(LOG_LEVEL) # Set the minimum level to capture

# Console Handler (prints to stderr/stdout, visible in `docker logs`)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(LOG_LEVEL) # Set level for console output
logger.addHandler(stream_handler)

# File Handler (writes to the specified log file)
try:
    log_file = Path(LOG_FILE_PATH)
    log_file.parent.mkdir(parents=True, exist_ok=True) # Create log directory if it doesn't exist
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG) # Log everything to the file
    logger.addHandler(file_handler)
    logger.info(f"Logging initialized. Console Level: {logging.getLevelName(LOG_LEVEL)}, File Path: '{LOG_FILE_PATH}', File Level: DEBUG")
except Exception as e:
    logger.error(f"Failed to configure file logging to '{LOG_FILE_PATH}': {e}", exc_info=True)
    # Continue without file logging if setup fails

# --- Validate Backend URL ---
try:
    parsed_backend_url = urlparse(YOUR_BACKEND_API_URL)
    if not parsed_backend_url.scheme or not parsed_backend_url.netloc:
        raise ValueError("BACKEND_API_URL must be a valid absolute URL (e.g., http://localhost:5000/v1)")
    BACKEND_BASE_URL = urlunparse((parsed_backend_url.scheme, parsed_backend_url.netloc, '', '', '', ''))
    logger.info(f"Backend Base URL validated: {BACKEND_BASE_URL}")
except ValueError as e:
    logger.critical(f"Invalid BACKEND_API_URL '{YOUR_BACKEND_API_URL}': {e}")
    BACKEND_BASE_URL = None # Indicate failure

# --- FastAPI App ---
app = FastAPI(
    title="OpenAI Compatible Wrapper (LangGraph, Role Transformation, Null Content Fix)",
    description="Uses LangGraph, ensures only system/user/assistant roles sent to backend, and fixes null content for assistant tool calls.",
)

# --- Shared HTTP Client ---
http_client = httpx.AsyncClient(timeout=300.0)

@app.on_event("startup")
async def startup_event():
     logger.info("Application startup.")
     if not BACKEND_BASE_URL:
         logger.warning("Application starting with invalid BACKEND_API_URL configuration.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown.")
    await http_client.aclose()

# === LangGraph Definition ===

# 1. Define State
class GraphState(TypedDict):
    initial_request: Dict[str, Any]
    response_format_type: Literal["text", "json_object"]
    json_schema: Optional[Dict[str, Any]]
    backend_payload: Optional[Dict[str, Any]]
    backend_response_data: Optional[Dict[str, Any]]
    backend_response_content: Optional[str]
    llm_call_status: Optional[Literal["success", "retryable_error", "fatal_error"]]
    parsed_tool_calls: Optional[List[Dict[str, Any]]]
    parsed_json_object: Optional[Dict[str, Any]]
    final_openai_response: Optional[Dict[str, Any]]
    error: Optional[str]
    retry_count: int
    max_retries: int

# 2. Define Helper functions for message preparation

def _transform_tool_role_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transforms messages with role "tool" into role "user" messages.
    """
    transformed_messages = []
    for msg in messages:
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown_id")
            content = msg.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content)
                    logger.warning(f"Tool role message content for call_id '{tool_call_id}' was not a string, converted from JSON.")
                except Exception:
                    content = str(content)
                    logger.warning(f"Tool role message content for call_id '{tool_call_id}' was not a string, converted using str(). Type was: {type(content)}")

            user_content = f"[Tool execution result for call {tool_call_id}]:\n{content}"
            transformed_messages.append({"role": "user", "content": user_content})
            logger.debug(f"Transformed 'tool' role (id: {tool_call_id}) to 'user' role for backend.")
        else:
            transformed_messages.append(msg.copy()) # Make a copy to be safe
    return transformed_messages

def _prepare_json_mode_messages(messages: List[Dict[str, Any]], json_schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ Prepares messages for JSON mode. First transforms tool roles, then adds JSON schema instructions. """
    current_messages = _transform_tool_role_messages(messages)
    modified_messages_with_json_prompt = [msg.copy() for msg in current_messages]
    schema_string = "{}"
    if isinstance(json_schema, dict):
        try: schema_string = json.dumps(json_schema, indent=2)
        except Exception as e:
            logger.warning(f"Could not serialize JSON schema for prompt: {e}")
            schema_string = "[Error displaying schema]"
    json_instructions = (
        "\n\nIMPORTANT: You MUST respond ONLY with a single, valid JSON object that strictly adheres "
        "to the following JSON Schema. Do NOT include any other text, explanations, apologies, or markdown formatting "
        "before or after the JSON object. Your entire response must be the JSON object itself.\n\n"
        "JSON Schema:\n"
        "```json\n"
        f"{schema_string}\n"
        "```\n"
        "Ensure your output is valid JSON and matches this schema precisely."
    )
    system_message_found = False
    for msg in modified_messages_with_json_prompt:
        if msg.get("role") == "system":
            msg["content"] = msg.get("content", "") + json_instructions
            system_message_found = True; break
    if not system_message_found:
        modified_messages_with_json_prompt.insert(0, {"role": "system", "content": json_instructions.strip()})
    return modified_messages_with_json_prompt

def _prepare_prompt_messages(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Prepares messages for text mode. First transforms tool roles,
    then potentially adds tool usage instructions if tools are provided for the current turn.
    """
    current_messages = _transform_tool_role_messages(messages)
    if not tools:
        return current_messages
    modified_messages_with_tool_prompt = [msg.copy() for msg in current_messages]
    try: tool_descriptions = "\n\nAvailable Tools:\n" + json.dumps(tools, indent=2)
    except Exception as e:
        logger.warning(f"Could not serialize tools for prompt: {e}")
        tool_descriptions = "\n\n[Error displaying available tools]"
    instructions = (
        "\n\nIf you determine you need to use one or more of the above tools to respond, "
        "please output ONLY a JSON object matching the following structure, "
        "with no other text before or after it. The 'tool_calls' field MUST be an array, "
        "containing one object for each tool you need to call:\n"
        "{\n"
        "  \"tool_calls\": [\n"
        "    {\n"
        "      \"name\": \"<name_of_tool_to_call>\",\n"
        "      \"arguments\": { <arguments_as_object> }\n"
        "    }\n"
        "    // , { ... more tool calls if needed ... }\n"
        "  ]\n"
        "}\n"
        "If you do not need to use a tool, respond naturally with your text answer."
    )
    system_message_found = False
    for msg in modified_messages_with_tool_prompt:
        if msg.get("role") == "system":
            msg["content"] = msg.get("content", "") + tool_descriptions + instructions
            system_message_found = True; break
    if not system_message_found:
        modified_messages_with_tool_prompt.insert(0, {"role": "system", "content": (tool_descriptions + instructions).strip()})
    return modified_messages_with_tool_prompt

# 3. Define Nodes

async def prepare_request(state: GraphState) -> Dict[str, Any]:
    """
    Prepares messages and payload for backend LLM.
    Transforms 'tool' roles to 'user'.
    Ensures assistant messages with tool_calls have "" for content if null.
    Initializes retry state and handles response_format.
    """
    logger.debug("--- Node: prepare_request ---")
    initial_request = state['initial_request']
    messages_from_client = initial_request.get("messages", [])
    model_name = initial_request.get("model", "default-model")
    tools_from_client = initial_request.get("tools")
    tool_choice_from_client = initial_request.get("tool_choice")
    response_format = initial_request.get("response_format", {"type": "text"})

    response_format_type: Literal["text", "json_object"] = "text"
    json_schema: Optional[Dict[str, Any]] = None
    current_error = None
    backend_messages: List[Dict[str, Any]] = []

    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        response_format_type = "json_object"
        json_schema = response_format.get("schema")
        if not isinstance(json_schema, dict):
            error_msg = "Invalid request: response_format type is 'json_object' requires a 'schema' dictionary."
            logger.error(f"prepare_request: {error_msg}")
            current_error = error_msg
            json_schema = None
        else:
            logger.debug(f"JSON Object mode requested with schema: {json.dumps(json_schema)}")
        backend_messages = _prepare_json_mode_messages(messages_from_client, json_schema)
    else:
        response_format_type = "text"
        logger.debug("Text mode requested (or default).")
        backend_messages = _prepare_prompt_messages(messages_from_client, tools_from_client)

    # Sanitize assistant messages: if content is null and tool_calls exist, set content to ""
    sanitized_backend_messages = []
    for msg_dict in backend_messages: # Iterate over list of dicts
        new_msg = msg_dict.copy() # Ensure it's a mutable copy
        if new_msg.get("role") == "assistant" and new_msg.get("tool_calls") and new_msg.get("content") is None:
            new_msg["content"] = ""
            logger.debug(f"Sanitized assistant message with tool_calls: changed null content to empty string.")
        sanitized_backend_messages.append(new_msg)
    backend_messages = sanitized_backend_messages # Replace with sanitized list

    backend_payload = None
    if not current_error:
        payload_extras = {k: v for k, v in initial_request.items() if k not in [
            "messages", "tools", "tool_choice", "model", "response_format", "stream", "max_retries"
        ]}
        backend_payload = {
            "model": model_name,
            "messages": backend_messages,
            **payload_extras,
            "stream": False,
        }
        # Note: For text mode with tool calling, we handle tools via prompts
        # Don't pass tools/tool_choice to backend if it doesn't support them
        # Comment out the following lines to use text-based tool calling:
        # if response_format_type == "text":
        #     if tools_from_client:
        #         backend_payload["tools"] = tools_from_client
        #     if tool_choice_from_client:
        #         backend_payload["tool_choice"] = tool_choice_from_client
        logger.info(f"Prepared backend payload for model: {model_name}, Format Mode: {response_format_type}")

    return {
        "response_format_type": response_format_type,
        "json_schema": json_schema,
        "backend_payload": backend_payload,
        "retry_count": 0,
        "max_retries": initial_request.get("max_retries", DEFAULT_MAX_RETRIES),
        "error": current_error,
        "llm_call_status": None,
        "backend_response_data": None,
        "backend_response_content": None,
        "parsed_tool_calls": None,
        "parsed_json_object": None,
        "final_openai_response": None,
    }

async def invoke_backend_llm(state: GraphState) -> Dict[str, Any]:
    if state.get('error'):
        logger.warning("invoke_backend_llm: Skipped due to previous error in state.")
        return {"error": state['error'], "backend_response_data": None, "backend_response_content": None}

    logger.info(f"--- Node: invoke_backend_llm (Attempt: {state['retry_count'] + 1}/{state['max_retries'] + 1}) ---")
    backend_payload = state.get('backend_payload')
    if not backend_payload:
        logger.critical("CRITICAL: Backend payload missing in state before call.")
        return {"error": "CRITICAL: Backend payload missing in state before call.", "backend_response_data": None, "backend_response_content": None}
    if not BACKEND_BASE_URL:
        error_msg = "Cannot invoke backend: BACKEND_API_URL was invalid during startup."
        logger.error(error_msg)
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}

    headers = {"Content-Type": "application/json"}
    if YOUR_BACKEND_API_KEY: headers["Authorization"] = f"Bearer {YOUR_BACKEND_API_KEY}"
    target_url = urljoin(BACKEND_BASE_URL, "/v1/chat/completions")

    logger.debug(f"Sending Request to Backend URL: {target_url}")
    logger.debug("--- Backend Request Payload ---")
    try: logger.debug(json.dumps(backend_payload, indent=2))
    except Exception as e: logger.debug(f"(Error pretty-printing payload: {e}) Payload: {backend_payload}")
    logger.debug("-----------------------------")

    backend_resp = None
    try:
        start_time = time.time()
        backend_req = http_client.build_request("POST", target_url, json=backend_payload, headers=headers)
        backend_resp = await http_client.send(backend_req, stream=False)
        end_time = time.time()
        logger.info(f"Backend call completed - Status: {backend_resp.status_code}, Time: {end_time - start_time:.2f}s")

        if backend_resp.status_code >= 400:
            error_detail_text = f"Backend API Error ({backend_resp.status_code})"
            logger.error(f"invoke_backend_llm: {error_detail_text} from URL {backend_resp.url}")
            logger.debug(f"Backend error response headers: {backend_resp.headers}")
            raw_body_bytes = b''
            try:
                raw_body_bytes = backend_resp.content # Access preloaded bytes
                if raw_body_bytes:
                    encoding = backend_resp.encoding or 'utf-8'
                    decoded_body = raw_body_bytes.decode(encoding, errors='replace')
                    error_detail_text += f": {decoded_body}"
                    logger.debug(f"Backend error response body (decoded): {decoded_body}")
                else:
                    logger.debug("Backend error response body is empty.")
            except Exception as e_read:
                logger.error(f"Failed to read/decode backend error response body: {e_read}. Raw byte length: {len(raw_body_bytes)}")
                error_detail_text += ": [Could not read/decode error body]"
            try: backend_data = backend_resp.json()
            except Exception: backend_data = None
            return {"error": error_detail_text, "backend_response_data": backend_data, "backend_response_content": None}

        backend_data = backend_resp.json()
        backend_content = None
        try:
            message = backend_data.get("choices", [{}])[0].get("message", {})
            backend_content = message.get("content") if message else None
            if message and message.get("tool_calls"):
                logger.debug("Backend response message contains native 'tool_calls'.")
            if state.get("response_format_type") == "json_object" and backend_content is None:
                 logger.warning("Backend returned null content, but JSON object was expected.")
        except (IndexError, TypeError, AttributeError) as parse_err:
             error_msg = f"Could not parse expected structure from backend response: {parse_err}"
             logger.error(f"invoke_backend_llm: {error_msg} \nData: {backend_data}")
             return {"error": error_msg, "backend_response_data": backend_data, "backend_response_content": None}

        logger.debug("Backend response parsed successfully.")
        logger.debug(f"Backend Raw Content Snippet: {str(backend_content)[:200]}...")
        return {"backend_response_data": backend_data, "backend_response_content": backend_content, "error": None}

    except httpx.TimeoutException as e:
        error_msg = f"Request to backend timed out: {e}"
        logger.error(f"invoke_backend_llm: {error_msg}")
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}
    except httpx.RequestError as e:
        error_msg = f"Could not connect/request backend: {e}"
        logger.error(f"invoke_backend_llm: {error_msg}")
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode backend JSON response: {e}"
        logger.error(f"invoke_backend_llm: {error_msg}")
        raw_text_from_error = ""
        if backend_resp:
            try: raw_text_from_error = backend_resp.text
            except Exception as read_err: logger.error(f"Could not get raw text from error response: {read_err}")
        logger.error(f"Raw response text (if available): {raw_text_from_error}")
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}
    except Exception as e:
        error_msg = f"An internal error occurred calling backend: {str(e)}"
        logger.exception(f"invoke_backend_llm: {error_msg}")
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}
    finally:
        if backend_resp and not backend_resp.is_closed:
            await backend_resp.aclose()

async def check_llm_call_status(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: check_llm_call_status ---")
    error = state.get('error')
    retry_count = state['retry_count']
    max_retries = state['max_retries']
    status: Literal["success", "retryable_error", "fatal_error"] = "success"
    if error:
        error_str = str(error).lower()
        if state.get('backend_payload') is None and ("invalid request" in error_str or "invalid during startup" in error_str):
            logger.debug("Status: fatal_error (Invalid request or config before LLM call)")
            status = "fatal_error"
        else:
            is_retryable = "timed out" in error_str or \
                           "could not connect" in error_str or \
                           "request to backend timed out" in error_str or \
                           "could not connect/request backend" in error_str or \
                           any(code in error_str for code in ["(500)", "(502)", "(503)", "(504)"])
            if is_retryable and retry_count < max_retries:
                 logger.debug(f"Status: retryable_error (Attempt {retry_count + 1}/{max_retries + 1})")
                 status = "retryable_error"
            else:
                 log_reason = "Non-retryable API/internal error" if not is_retryable else f"max retries ({max_retries + 1}) reached"
                 logger.debug(f"Status: fatal_error ({log_reason}: {error})")
                 status = "fatal_error"
    else:
         logger.debug("Status: success")
    return {"llm_call_status": status}

async def wait_and_retry(state: GraphState) -> Dict[str, Any]:
    retry_count = state['retry_count'] + 1
    error_msg = state.get('error', 'Unknown error triggering retry')
    logger.warning(f"--- Node: wait_and_retry ---")
    logger.warning(f"Retry attempt {retry_count}/{state['max_retries'] + 1} due to error: {error_msg}")
    delay = 1 * (2 ** (retry_count - 1))
    delay = min(delay, 30)
    logger.info(f"Waiting for {delay:.2f} seconds before retrying...")
    await asyncio.sleep(delay)
    return {"retry_count": retry_count, "error": None, "llm_call_status": None}

async def route_by_format(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: route_by_format ---")
    return {}

async def parse_and_validate_structured_json(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: parse_and_validate_structured_json ---")
    backend_content = state.get('backend_response_content')
    json_schema = state.get('json_schema')
    current_error = None
    parsed_object = None
    if backend_content is None:
        current_error = "Validation failed: Backend response content is missing (None)."
        logger.warning(current_error)
    elif not isinstance(backend_content, str) or not backend_content.strip():
         current_error = f"Validation failed: Backend response content is not a non-empty string or is empty. Type: {type(backend_content)}"
         logger.warning(current_error)
    elif not json_schema:
        current_error = "Validation failed: JSON schema is missing in state."
        logger.error(current_error)

    if not current_error:
        processed_content = backend_content
        think_tag_pattern = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
        original_length = len(processed_content)
        processed_content = think_tag_pattern.sub("", processed_content)
        if len(processed_content) < original_length: logger.debug("Stripped <think> tags.")
        content_to_parse = processed_content.strip()
        fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
        match = fence_pattern.search(content_to_parse)
        original_content_was_fenced = False
        if match:
            logger.debug("Stripped markdown fences from JSON content.")
            content_to_parse = match.group(1).strip()
            original_content_was_fenced = True
        elif not (content_to_parse.startswith("{") and content_to_parse.endswith("}")):
             logger.debug("Content does not appear to be JSON or fenced JSON after stripping think tags.")
        if not content_to_parse:
            current_error = "Validation failed: Content became empty after stripping."
            logger.warning(current_error)
        else:
            try:
                parsed_object = json.loads(content_to_parse)
                logger.debug("Successfully parsed backend content as JSON.")
                try:
                    jsonschema.validate(instance=parsed_object, schema=json_schema)
                    logger.debug("JSON object successfully validated against schema.")
                except ValidationError as e:
                    current_error = (f"Validation failed: JSON does not match schema. Error: {e.message} "
                                     f"on path: {'/'.join(map(str, e.path))}. JSON: {str(parsed_object)[:200]}...")
                    logger.error(current_error)
                except Exception as e: # Other jsonschema errors
                    current_error = f"Validation error: {e}"
                    logger.exception(current_error)
            except json.JSONDecodeError as e:
                error_detail = f"Error: {e}." + (" Content was fenced." if original_content_was_fenced else "")
                current_error = f"Validation failed: Not valid JSON after stripping. {error_detail} Content: '{content_to_parse[:200]}...'"
                logger.error(current_error)
    if current_error: return {"error": current_error, "parsed_json_object": parsed_object}
    return {"parsed_json_object": parsed_object, "error": None}

async def check_text_for_tool_calls(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: check_text_for_tool_calls ---")
    backend_content = state.get('backend_response_content')
    parsed_tool_calls = None; error = None
    if backend_content is None:
        logger.warning("Backend content is None in check_text_for_tool_calls.")
    elif isinstance(backend_content, str):
        detected_tool_calls, _ = _parse_backend_response_for_tool_calls(backend_content)
        if detected_tool_calls:
            logger.debug(f"Parsed {len(detected_tool_calls)} tool call(s).")
            parsed_tool_calls = detected_tool_calls
        else: logger.debug("Parsed 0 tool calls (Text response).")
    else:
        error = f"Parsing failed: Expected string content, got {type(backend_content)}"
        logger.error(error)
    return {"parsed_tool_calls": parsed_tool_calls, "error": error}

async def format_structured_json_response(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: format_structured_json_response ---")
    parsed_json_object = state.get('parsed_json_object')
    backend_data = state.get('backend_response_data')
    initial_request = state['initial_request']
    current_error = None; final_response_payload = None
    if parsed_json_object is None: current_error = "Cannot format: Validated JSON object missing."
    elif backend_data is None: current_error = "Cannot format: Backend response data missing."
    if current_error: logger.error(current_error)
    else:
        try:
            usage = backend_data.get("usage", {})
            model = backend_data.get("model", initial_request.get("model", "unknown"))
            content_str = json.dumps(parsed_json_object)
            msg = {"role": "assistant", "content": content_str, "tool_calls": None}
            final_response_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": "stop"}], "usage": usage,
            }
            logger.debug("Formatted JSON object response for client.")
        except Exception as e:
            current_error = f"Error formatting JSON response: {e}"
            logger.exception(current_error)
    return {"final_openai_response": final_response_payload, "error": current_error}

async def format_tool_calls_response(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: format_tool_calls_response ---")
    parsed_tool_calls = state.get('parsed_tool_calls')
    backend_data = state.get('backend_response_data')
    initial_request = state['initial_request']
    current_error = None; final_response_payload = None
    if not parsed_tool_calls: current_error = "Cannot format: Parsed tool calls missing."
    elif not backend_data: current_error = "Cannot format: Backend response data missing."
    if current_error: logger.error(current_error)
    else:
        try:
            usage = backend_data.get("usage", {})
            model = backend_data.get("model", initial_request.get("model", "unknown"))
            openai_tools = []
            for tc_info in parsed_tool_calls:
                args_str = json.dumps(tc_info.get("arguments", {}))
                openai_tools.append({"id": f"call_{uuid.uuid4().hex}", "type": "function",
                                     "function": {"name": tc_info["name"], "arguments": args_str}})
            msg = {"role": "assistant", "content": None, "tool_calls": openai_tools}
            final_response_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": "tool_calls"}], "usage": usage,
            }
            logger.debug("Formatted tool call response for client.")
        except Exception as e:
            current_error = f"Error formatting tool call response: {e}"
            logger.exception(current_error)
    return {"final_openai_response": final_response_payload, "error": current_error}

async def format_plain_text_response(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: format_plain_text_response ---")
    content = state.get('backend_response_content', "")
    backend_data = state.get('backend_response_data')
    initial_request = state['initial_request']
    current_error = None; final_response_payload = None
    if backend_data is None: current_error = "Cannot format: Backend response data missing."
    if current_error: logger.error(current_error)
    else:
        try:
            usage = backend_data.get("usage", {})
            finish_reason = backend_data.get("choices", [{}])[0].get("finish_reason", "stop")
            model = backend_data.get("model", initial_request.get("model", "unknown"))
            msg = {"role": "assistant", "content": content, "tool_calls": None}
            final_response_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": msg, "logprobs": None, "finish_reason": finish_reason}], "usage": usage,
            }
            logger.debug("Formatted text response for client.")
        except Exception as e:
            current_error = f"Error formatting text response: {e}"
            logger.exception(current_error)
    return {"final_openai_response": final_response_payload, "error": current_error}

async def format_error_response(state: GraphState) -> Dict[str, Any]:
    logger.debug("--- Node: format_error_response ---")
    error_msg = state.get("error", "Unknown error during processing.")
    logger.error(f"Preparing final error response: {error_msg}")
    payload = {"object": "error", "message": str(error_msg), "type": "api_error", "param": None, "code": None}
    return {"final_openai_response": payload, "error": error_msg or "Unknown error"}

def _parse_backend_response_for_tool_calls(backend_response_content: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if not isinstance(backend_response_content, str): return None, backend_response_content
    content_trimmed = backend_response_content.strip()
    match = re.search(r"^(?:```(?:json)?\s*)?(\{[\s\n]*\"tool_calls\":\s*\[.*?\][\s\n]*\})(?:\s*```)?$", content_trimmed, re.DOTALL | re.IGNORECASE)
    if not match: return None, backend_response_content
    potential_json_str = match.group(1)
    try:
        data = json.loads(potential_json_str)
        if isinstance(data, dict) and "tool_calls" in data:
            tool_calls_list = data["tool_calls"]
            if isinstance(tool_calls_list, list) and all(isinstance(call, dict) and "name" in call and "arguments" in call for call in tool_calls_list):
                if not tool_calls_list:
                    logger.debug("Parsed empty 'tool_calls' array from backend.")
                    return None, backend_response_content
                valid_calls = []
                for call in tool_calls_list:
                    args = call.get("arguments"); parsed_args = None
                    if isinstance(args, dict): parsed_args = args
                    elif isinstance(args, str):
                        try: parsed_args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.warning(f"Tool '{call.get('name')}' args string not JSON: '{args}'. Using raw.")
                            parsed_args = args
                    else:
                        logger.warning(f"Tool '{call.get('name')}' args type {type(args)}. Using empty dict.")
                        parsed_args = {}
                    valid_calls.append({"name": call["name"], "arguments": parsed_args})
                logger.debug(f"Successfully parsed {len(valid_calls)} tool calls from backend content.")
                return valid_calls, backend_response_content
        return None, backend_response_content
    except json.JSONDecodeError:
        logger.debug(f"Failed to decode potential tool call JSON: '{potential_json_str}'")
        return None, backend_response_content

# 4. Define Conditional Edge Logic
def route_on_llm_status(state: GraphState) -> Literal["wait_and_retry", "route_by_format", "format_error_response"]:
    status = state.get("llm_call_status")
    logger.debug(f"--- Decision: route_on_llm_status (Status: {status}) ---")
    if status == "retryable_error": return "wait_and_retry"
    if status == "success": return "route_by_format"
    return "format_error_response"

def route_on_response_format(state: GraphState) -> Literal["parse_and_validate_structured_json", "check_text_for_tool_calls"]:
    fmt = state.get("response_format_type", "text")
    logger.debug(f"--- Decision: route_on_response_format (Format: {fmt}) ---")
    if fmt == "json_object": return "parse_and_validate_structured_json"
    return "check_text_for_tool_calls"

def check_json_validation_result(state: GraphState) -> Literal["format_structured_json_response", "format_error_response"]:
    error = state.get('error'); parsed_json = state.get('parsed_json_object')
    logger.debug(f"--- Decision: check_json_validation_result (Error: {error}, Parsed: {parsed_json is not None}) ---")
    if error or parsed_json is None:
        if not error: state['error'] = "JSON validation failed: No object parsed."
        return "format_error_response"
    return "format_structured_json_response"

def route_text_mode_result(state: GraphState) -> Literal["format_tool_calls_response", "format_plain_text_response", "format_error_response"]:
    error = state.get('error'); parsed_tool_calls = state.get('parsed_tool_calls')
    logger.debug(f"--- Decision: route_text_mode_result (Error: {error}, Tools: {parsed_tool_calls is not None}) ---")
    if error: return "format_error_response"
    if parsed_tool_calls: return "format_tool_calls_response"
    return "format_plain_text_response"

# 5. Build the Graph
workflow = StateGraph(GraphState)
workflow.add_node("prepare_request", prepare_request)
workflow.add_node("invoke_backend_llm", invoke_backend_llm)
workflow.add_node("check_llm_call_status", check_llm_call_status)
workflow.add_node("wait_and_retry", wait_and_retry)
workflow.add_node("route_by_format", route_by_format)
workflow.add_node("parse_and_validate_structured_json", parse_and_validate_structured_json)
workflow.add_node("format_structured_json_response", format_structured_json_response)
workflow.add_node("check_text_for_tool_calls", check_text_for_tool_calls)
workflow.add_node("format_tool_calls_response", format_tool_calls_response)
workflow.add_node("format_plain_text_response", format_plain_text_response)
workflow.add_node("format_error_response", format_error_response)

workflow.set_entry_point("prepare_request")
workflow.add_edge("prepare_request", "invoke_backend_llm")
workflow.add_edge("invoke_backend_llm", "check_llm_call_status")
workflow.add_conditional_edges("check_llm_call_status", route_on_llm_status,
    {"wait_and_retry": "wait_and_retry", "route_by_format": "route_by_format", "format_error_response": "format_error_response"})
workflow.add_edge("wait_and_retry", "invoke_backend_llm")
workflow.add_conditional_edges("route_by_format", route_on_response_format,
    {"parse_and_validate_structured_json": "parse_and_validate_structured_json", "check_text_for_tool_calls": "check_text_for_tool_calls"})
workflow.add_conditional_edges("parse_and_validate_structured_json", check_json_validation_result,
    {"format_structured_json_response": "format_structured_json_response", "format_error_response": "format_error_response"})
workflow.add_conditional_edges("check_text_for_tool_calls", route_text_mode_result,
    {"format_tool_calls_response": "format_tool_calls_response", "format_plain_text_response": "format_plain_text_response", "format_error_response": "format_error_response"})
workflow.add_edge("format_structured_json_response", END)
workflow.add_edge("format_tool_calls_response", END)
workflow.add_edge("format_plain_text_response", END)
workflow.add_edge("format_error_response", END)

# 6. Compile the Graph
try:
    langgraph_app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.")
except Exception as e:
    logger.critical(f"FATAL ERROR: LangGraph workflow compilation failed: {e}", exc_info=True)
    raise e

try:
    if 'langgraph_app' in locals():
        try:
            import pygraphviz # type: ignore
            png_bytes = langgraph_app.get_graph(xray=True).draw_mermaid_png()
            graph_filename = "flow.png"
            with open(graph_filename, "wb") as f: f.write(png_bytes)
            logger.info(f"Saved refactored graph visualization to {graph_filename}")
        except ImportError:
            logger.warning("Skipping graph visualization: `pygraphviz` or `graphviz` (system library) not installed.")
        except Exception as e_draw:
            logger.error(f"Could not draw graph: {e_draw}", exc_info=True)
    else:
        logger.warning("Skipping graph visualization because LangGraph compilation failed.")
except Exception as e_graph_outer:
    logger.error(f"An unexpected error occurred during graph visualization setup: {e_graph_outer}", exc_info=True)

# === FastAPI Endpoints ===
@app.post("/v1/chat/completions")
async def chat_completions_langgraph_proxy(request: Request):
    request_id = uuid.uuid4()
    logger.info(f"Received request {request_id} for /v1/chat/completions from {request.client.host}")
    try:
        openai_payload = await request.json()
        logger.debug(f"Request {request_id} Payload: {json.dumps(openai_payload, indent=2)}")
    except json.JSONDecodeError:
        logger.error(f"Request {request_id}: Invalid JSON payload.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    if openai_payload.get("stream", False):
        logger.warning(f"Request {request_id}: Streaming requested, but LangGraph is non-streaming. Returning full response.")

    initial_state: GraphState = {
        "initial_request": openai_payload, "response_format_type": "text", "json_schema": None,
        "backend_payload": None, "backend_response_data": None, "backend_response_content": None,
        "llm_call_status": None, "parsed_tool_calls": None, "parsed_json_object": None,
        "final_openai_response": None, "error": None, "retry_count": 0,
        "max_retries": openai_payload.get("max_retries", DEFAULT_MAX_RETRIES)
    }

    logger.info(f"Invoking LangGraph for request {request_id}...")
    start_time = time.time()
    final_state = None
    try:
        final_state = await langgraph_app.ainvoke(initial_state)
        end_time = time.time()
        logger.info(f"LangGraph execution for {request_id} finished in {end_time - start_time:.2f}s")
    except Exception as graph_exec_error:
        logger.critical(f"CRITICAL: Unhandled LangGraph exception for {request_id}: {graph_exec_error}", exc_info=True)
        err_payload = {"object": "error", "message": "Internal server error during request processing.", "type": "api_error"}
        return JSONResponse(content=err_payload, status_code=500)

    if not isinstance(final_state, dict):
        logger.critical(f"CRITICAL: LangGraph returned non-dict state for {request_id}: {type(final_state)}")
        err_payload = {"object": "error", "message": "Internal server error: Invalid graph final state.", "type": "api_error"}
        return JSONResponse(content=err_payload, status_code=500)

    logger.debug(f"--- LangGraph Final State ({request_id}) ---")
    for key, value in final_state.items(): # Log final state details
        if key == "final_openai_response" and value:
            try: logger.debug(f"  {key}: {json.dumps(value, indent=2)}")
            except: logger.debug(f"  {key} (type {type(value)}): {str(value)[:500]}...") # Truncate if large/unserializable
        elif isinstance(value, (dict, list)):
             try: logger.debug(f"  {key}: {json.dumps(value, indent=2)}")
             except: logger.debug(f"  {key} (type {type(value)}): {str(value)[:200]}...")
        else:
            logger.debug(f"  {key}: {value}")
    logger.debug("---------------------------")


    final_response = final_state.get("final_openai_response")
    if final_response and isinstance(final_response, dict) and final_response.get("object") == "error":
         logger.error(f"Graph error for {request_id}, returning formatted error.")
         status_code = 500
         err_msg_lower = str(final_state.get('error', final_response.get("message", ""))).lower()
         if "backend api error" in err_msg_lower:
             match = re.search(r'\((\d{3})\)', err_msg_lower)
             status_code = int(match.group(1)) if match else 502
         elif any(k in err_msg_lower for k in ["timed out", "timeout"]): status_code = 504
         elif any(k in err_msg_lower for k in ["could not connect", "connection error"]): status_code = 503
         elif any(k in err_msg_lower for k in ["validation failed", "invalid request", "schema"]): status_code = 400
         logger.info(f"{request_id} returning error - Status: {status_code}, Message: {final_response.get('message')}")
         return JSONResponse(content=final_response, status_code=status_code)

    if not final_response:
        err_detail = final_state.get("error", "Unknown processing error, no final response.")
        logger.critical(f"CRITICAL: No final response for {request_id}. Error: {err_detail}")
        err_payload = {"object": "error", "message": f"Internal error: No response. Detail: {err_detail}", "type": "api_error"}
        return JSONResponse(content=err_payload, status_code=500)

    logger.info(f"{request_id} successful. Sending Final Response from LangGraph.")
    return JSONResponse(content=final_response)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def generic_proxy(request: Request, path: str):
    if not BACKEND_BASE_URL:
        logger.error(f"Generic proxy error for '/{path}': BACKEND_API_URL invalid.")
        raise HTTPException(status_code=503, detail="Backend service misconfigured")

    request_id = uuid.uuid4()
    logger.info(f"Generic proxy request {request_id}: Path=/{path}, Method={request.method}")
    query_path = path if path.startswith('/') else f'/{path}'
    target_url = f"{BACKEND_BASE_URL.rstrip('/')}{query_path}"
    logger.debug(f"Request {request_id} targeting backend: {target_url}")

    headers = {h: v for h, v in request.headers.items() if h.lower() not in ['host', 'content-length', 'connection']}
    if YOUR_BACKEND_API_KEY: headers.setdefault("Authorization", f"Bearer {YOUR_BACKEND_API_KEY}")
    headers.setdefault('Connection', 'keep-alive')
    body = await request.body()
    backend_req = http_client.build_request(request.method, target_url, headers=headers, content=body, params=request.query_params)
    backend_resp = None
    try:
        start_time = time.time()
        backend_resp = await http_client.send(backend_req, stream=True)
        end_time = time.time()
        logger.info(f"Generic proxy {request_id} to {query_path} -> Backend Status: {backend_resp.status_code}, Time: {end_time - start_time:.2f}s")
        resp_headers = {h:v for h,v in backend_resp.headers.items() if h.lower() not in ['content-encoding','transfer-encoding','connection']}
        if 'content-type' in backend_resp.headers: resp_headers['content-type'] = backend_resp.headers['content-type']

        async def stream_gen():
            try:
                async for chunk in backend_resp.aiter_bytes(): yield chunk
                logger.debug(f"Generic proxy stream finished for {request_id} ({query_path})")
            finally:
                 if backend_resp and not backend_resp.is_closed: await backend_resp.aclose()
        return StreamingResponse(stream_gen(), status_code=backend_resp.status_code, headers=resp_headers, media_type=backend_resp.headers.get("content-type"))
    except httpx.TimeoutException:
        logger.error(f"Timeout proxying {request_id} to {query_path}.")
        if backend_resp: await backend_resp.aclose()
        raise HTTPException(status_code=504, detail="Backend timeout")
    except httpx.RequestError as e:
        logger.error(f"Connection/Request error proxying {request_id} to {query_path}: {e}")
        if backend_resp: await backend_resp.aclose()
        raise HTTPException(status_code=503, detail=f"Backend connection error: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in generic proxy for {request_id} ({query_path}): {e}")
        if backend_resp: await backend_resp.aclose()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

@app.get("/health")
async def health_check():
    logger.debug("Health check.")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    logger.info(f"--- Starting OpenAI Wrapper Server (LangGraph, Role Transform, Null Content Fix) ---")
    logger.info(f"Port: {WRAPPER_PORT}, Backend Target: {YOUR_BACKEND_API_URL}")
    logger.info(f"Parsed Backend Base: {BACKEND_BASE_URL or 'INVALID'}")
    logger.info(f"Default Retries: {DEFAULT_MAX_RETRIES}, Console Log Level: {logging.getLevelName(LOG_LEVEL)}")
    logger.info(f"Log File: {LOG_FILE_PATH}")
    if not BACKEND_BASE_URL:
        logger.critical("CRITICAL: BACKEND_API_URL not configured correctly. Proxy/LLM calls will fail.")
    uvicorn.run(f"{script_name}:app", host="0.0.0.0", port=WRAPPER_PORT, log_level=logging.getLevelName(LOG_LEVEL).lower(), reload=False)