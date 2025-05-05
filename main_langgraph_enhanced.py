# main_logged.py
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
    # Depending on severity, you might want to exit or raise here
    # For now, we log critically and let the app try to start.
    BACKEND_BASE_URL = None # Indicate failure

# --- FastAPI App ---
app = FastAPI(
    title="OpenAI Compatible Wrapper (Refactored LangGraph + Structured Output + Think Tags + File Logging)",
    description="Uses LangGraph with clearer node structure, conditional logic, retries, structured JSON output (handling think tags) for /v1/chat/completions. Logs to console and file.",
)

# --- Shared HTTP Client ---
# Increased timeout for potentially complex JSON generation
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

# 1. Define State (Enhanced with Structured Output support)
class GraphState(TypedDict):
    # ... (State definition remains the same) ...
    initial_request: Dict[str, Any]
    response_format_type: Literal["text", "json_object"] # Requested format type
    json_schema: Optional[Dict[str, Any]] # Schema if json_object is requested
    backend_payload: Optional[Dict[str, Any]]
    backend_response_data: Optional[Dict[str, Any]] # Raw JSON data from backend
    backend_response_content: Optional[str] # Raw text content from backend choice
    # Status after LLM call: 'success', 'retryable_error', 'fatal_error'
    llm_call_status: Optional[Literal["success", "retryable_error", "fatal_error"]]
    parsed_tool_calls: Optional[List[Dict[str, Any]]] # Parsed tool calls if detected (text mode)
    parsed_json_object: Optional[Dict[str, Any]] # Parsed & validated JSON object (json_object mode)
    final_openai_response: Optional[Dict[str, Any]] # Final formatted response for client
    error: Optional[str] # Error message if something went wrong
    retry_count: int # Current retry attempt number
    max_retries: int # Maximum allowed retries for this request

# 2. Define Nodes (Replace print with logger)

async def prepare_request(state: GraphState) -> Dict[str, Any]:
    """ Prepares the messages and payload for the backend LLM. Initializes retry state and handles response_format. """
    logger.debug("--- Node: prepare_request ---") # Use debug for verbose node entries
    initial_request = state['initial_request']
    messages = initial_request.get("messages", [])
    model_name = initial_request.get("model", "default-model")
    tools = initial_request.get("tools")
    response_format = initial_request.get("response_format", {"type": "text"})

    response_format_type: Literal["text", "json_object"] = "text"
    json_schema: Optional[Dict[str, Any]] = None
    current_error = None # Track errors within this node

    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        response_format_type = "json_object"
        json_schema = response_format.get("schema")
        if not isinstance(json_schema, dict):
            error_msg = "Invalid request: response_format type is 'json_object' requires a 'schema' dictionary."
            logger.error(f"prepare_request: {error_msg}") # Log error
            current_error = error_msg
            json_schema = None # Ensure schema is None if invalid
        else:
             logger.debug(f"JSON Object mode requested with schema: {json.dumps(json_schema)}")
        # Prepare messages based on mode, even if schema was invalid (error will halt later)
        backend_messages = _prepare_json_mode_messages(messages, json_schema) # Pass schema here
        tools = None # Disable standard tool processing/prompting if JSON mode is forced
    else:
        response_format_type = "text"
        logger.debug("Text mode requested (or default).")
        backend_messages = _prepare_prompt_messages(messages, tools)

    # Only create payload if no critical error occurred yet
    backend_payload = None
    if not current_error:
        backend_payload = {
            "model": model_name,
            "messages": backend_messages,
            **{k: v for k, v in initial_request.items() if k not in [
                "messages", "tools", "tool_choice", "model", "response_format", "stream", "max_retries"
            ]},
            "stream": False,
        }
        logger.info(f"Prepared backend payload for model: {model_name}, Format Mode: {response_format_type}") # Info level for payload prep

    # Initialize state for this run
    return {
        "response_format_type": response_format_type,
        "json_schema": json_schema,
        "backend_payload": backend_payload, # Will be None if error occurred
        "retry_count": 0,
        "max_retries": initial_request.get("max_retries", DEFAULT_MAX_RETRIES),
        "error": current_error, # Set error state if schema was invalid
        # Reset other fields
        "llm_call_status": None,
        "backend_response_data": None,
        "backend_response_content": None,
        "parsed_tool_calls": None,
        "parsed_json_object": None,
        "final_openai_response": None,
    }

async def invoke_backend_llm(state: GraphState) -> Dict[str, Any]:
    """ Calls the actual backend LLM API. Sets error state on failure but does not decide retry/fatal yet. """
    # Check if prepare_request already set an error (e.g., bad schema)
    if state.get('error'):
        logger.warning("invoke_backend_llm: Skipped due to previous error in state.") # Use warning if skipped
        return {
            "error": state['error'],
            "backend_response_data": None,
            "backend_response_content": None
        }

    logger.info(f"--- Node: invoke_backend_llm (Attempt: {state['retry_count'] + 1}/{state['max_retries'] + 1}) ---") # Info level for attempts
    backend_payload = state.get('backend_payload')
    # This check is redundant now due to the error check above, but safe to keep
    if not backend_payload:
        logger.critical("CRITICAL: Backend payload missing in state before call.") # Use critical for unexpected missing data
        return {"error": "CRITICAL: Backend payload missing in state before call.", "backend_response_data": None, "backend_response_content": None}
    # Check if BACKEND_BASE_URL was validated
    if not BACKEND_BASE_URL:
        error_msg = "Cannot invoke backend: BACKEND_API_URL was invalid during startup."
        logger.error(error_msg)
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}

    headers = {"Content-Type": "application/json"}
    if YOUR_BACKEND_API_KEY: headers["Authorization"] = f"Bearer {YOUR_BACKEND_API_KEY}"
    target_url = urljoin(BACKEND_BASE_URL, "/v1/chat/completions") # Uses validated base URL

    logger.debug(f"Sending Request to Backend URL: {target_url}")
    logger.debug("--- Backend Request Payload ---")
    try: logger.debug(json.dumps(backend_payload, indent=2))
    except Exception as e: logger.debug(f"(Error pretty-printing payload: {e}) Payload: {backend_payload}")
    logger.debug("-----------------------------")

    try:
        start_time = time.time()
        backend_req = http_client.build_request("POST", target_url, json=backend_payload, headers=headers)
        backend_resp = await http_client.send(backend_req, stream=False)
        end_time = time.time()
        logger.info(f"Backend call completed - Status: {backend_resp.status_code}, Time: {end_time - start_time:.2f}s") # Info level for call results

        if backend_resp.status_code >= 400:
            error_text = await backend_resp.atext()
            error_msg = f"Backend API Error ({backend_resp.status_code}): {error_text}"
            logger.error(f"invoke_backend_llm: {error_msg}") # Error level for API errors
            # Attempt to parse error response JSON if possible, otherwise keep as None
            try: backend_data = backend_resp.json()
            except Exception: backend_data = None
            # Set error state, next node will classify it
            return {"error": error_msg, "backend_response_data": backend_data, "backend_response_content": None}

        # Success case
        backend_data = backend_resp.json()
        backend_content = None
        try:
            message = backend_data.get("choices", [{}])[0].get("message", {})
            backend_content = message.get("content") if message else None
            if state.get("response_format_type") == "json_object" and backend_content is None:
                 logger.warning("Backend returned null content, but JSON object was expected.") # Warning for unexpected null
        except (IndexError, TypeError, AttributeError) as parse_err:
             error_msg = f"Could not parse expected structure (choices/message/content) from backend response: {parse_err}"
             logger.error(f"invoke_backend_llm: {error_msg}") # Error level for parsing errors
             logger.error("--- Problematic Backend Response Data ---")
             try: logger.error(json.dumps(backend_data, indent=2))
             except Exception as e: logger.error(f"(Error pretty-printing response data: {e}) Data: {backend_data}")
             logger.error("---------------------------------------")
             # Return error but keep data for context, next node will classify
             return {"error": error_msg, "backend_response_data": backend_data, "backend_response_content": None}

        logger.debug("Backend response parsed successfully.")
        logger.debug("--- Backend Response Data Snippet ---")
        try: logger.debug(json.dumps(backend_data, indent=2, default=str)[:1000] + "...")
        except Exception as e: logger.debug(f"(Error pretty-printing response data: {e})")
        logger.debug("-----------------------------------")
        if backend_content is not None: logger.debug(f"Backend Raw Content Snippet: {str(backend_content)[:200]}...")
        else: logger.debug("Backend Raw Content: None")

        # Success: Clear error state
        return {
            "backend_response_data": backend_data,
            "backend_response_content": backend_content,
            "error": None
        }

    # --- Network/Timeout Errors ---
    except httpx.TimeoutException as e:
        error_msg = f"Request to backend timed out: {e}"
        logger.error(f"invoke_backend_llm: {error_msg}") # Error level
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}
    except httpx.RequestError as e:
        error_msg = f"Could not connect to backend: {e}"
        logger.error(f"invoke_backend_llm: {error_msg}") # Error level
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}
    # --- Other Errors ---
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode backend JSON response: {e}"
        logger.error(f"invoke_backend_llm: {error_msg}") # Error level
        raw_text = None
        if 'backend_resp' in locals():
            try: raw_text = await backend_resp.atext()
            except Exception: pass
        logger.error(f"Raw response text (if available): {raw_text}")
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}
    except Exception as e:
        error_msg = f"An internal error occurred calling backend: {str(e)}"
        logger.exception(f"invoke_backend_llm: {error_msg}") # Use logger.exception to include traceback
        return {"error": error_msg, "backend_response_data": None, "backend_response_content": None}

# --- NEW Node ---
async def check_llm_call_status(state: GraphState) -> Dict[str, Any]:
    """ Classifies the outcome of the LLM call into 'success', 'retryable_error', or 'fatal_error'. """
    logger.debug("--- Node: check_llm_call_status ---")
    error = state.get('error')
    retry_count = state['retry_count']
    max_retries = state['max_retries']
    status: Literal["success", "retryable_error", "fatal_error"] = "success" # Assume success initially

    if error:
        error_str = str(error)
        # Check for pre-call errors (e.g., invalid schema)
        if state.get('backend_payload') is None and ("Invalid request" in error_str or "invalid during startup" in error_str):
            logger.debug("Status: fatal_error (Invalid request or config before LLM call)")
            status = "fatal_error"
        else:
            # Check for retryable errors from the LLM call itself
            is_retryable = "timed out" in error_str or \
                           "Could not connect" in error_str or \
                           any(code in error_str for code in ["(500)", "(502)", "(503)", "(504)"])

            if is_retryable and retry_count < max_retries:
                 logger.debug(f"Status: retryable_error (Attempt {retry_count + 1}/{max_retries + 1})")
                 status = "retryable_error"
            else:
                 log_reason = "Non-retryable" if not is_retryable else f"max retries {max_retries + 1} reached"
                 logger.debug(f"Status: fatal_error ({log_reason})")
                 status = "fatal_error"
    else:
         logger.debug("Status: success")
         status = "success"

    return {"llm_call_status": status} # Update state with the classified status

async def wait_and_retry(state: GraphState) -> Dict[str, Any]:
    """ Increments retry count and introduces a small delay before looping back. """
    retry_count = state['retry_count'] + 1
    error_msg = state.get('error', 'Unknown error triggering retry')
    logger.warning(f"--- Node: wait_and_retry ---") # Warning level for retries
    logger.warning(f"Retry attempt {retry_count}/{state['max_retries'] + 1} due to error: {error_msg}")
    delay = 1 * (2 ** (retry_count - 1)) # Exponential backoff (1s, 2s, 4s...)
    delay = min(delay, 30) # Cap delay at 30 seconds
    logger.info(f"Waiting for {delay:.2f} seconds before retrying...")
    await asyncio.sleep(delay)
    # Clear the error for the next attempt, update retry count
    # Clear the status field as well
    return {"retry_count": retry_count, "error": None, "llm_call_status": None}

# --- NEW Node (Placeholder for routing) ---
async def route_by_format(state: GraphState) -> Dict[str, Any]:
    """ Simple node acting as a source for format-based routing after successful LLM call. """
    logger.debug("--- Node: route_by_format ---")
    # This node doesn't need to modify the state, just acts as a branching point.
    return {}

async def parse_and_validate_structured_json(state: GraphState) -> Dict[str, Any]:
    """ Parses backend content as JSON and validates against the schema.
        Includes stripping of markdown fences and <think> tags.
    """
    logger.debug("--- Node: parse_and_validate_structured_json ---")
    backend_content = state.get('backend_response_content')
    json_schema = state.get('json_schema')
    current_error = None # Track errors locally
    parsed_object = None # Initialize parsed_object

    if backend_content is None:
        current_error = "Validation failed: Backend response content is missing (None)."
        logger.warning(f"parse_and_validate_structured_json: {current_error}") # Warning if content is missing
    elif not isinstance(backend_content, str) or not backend_content.strip():
         current_error = f"Validation failed: Backend response content is not a non-empty string or is empty. Type: {type(backend_content)}"
         logger.warning(f"parse_and_validate_structured_json: {current_error}") # Warning for wrong type/empty
    elif not json_schema: # Should have been caught earlier, but double-check
        current_error = "Validation failed: JSON schema is missing in state."
        logger.error(f"parse_and_validate_structured_json: {current_error}") # Error if schema is missing here

    if not current_error:
        processed_content = backend_content # Start with original content

        # --- FIX 1: Strip <think>...</think> tags ---
        think_tag_pattern = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
        original_length = len(processed_content)
        processed_content = think_tag_pattern.sub("", processed_content)
        if len(processed_content) < original_length:
             logger.debug("Stripped <think> tags from content.")
        # --- End Fix 1 ---

        # --- FIX 2: Strip markdown fences and whitespace ---
        content_to_parse = processed_content.strip()
        fence_pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
        match = fence_pattern.search(content_to_parse)
        original_content_was_fenced = False
        if match:
            logger.debug("Stripped markdown fences from JSON content.")
            content_to_parse = match.group(1).strip() # Extract content within fences and strip again
            original_content_was_fenced = True
        elif content_to_parse.startswith("{") and content_to_parse.endswith("}"):
             logger.debug("Content appears to be JSON without fences.")
             pass # Assume it might be JSON without fences, proceed
        else:
             logger.debug("Content does not appear to be JSON or fenced JSON after stripping think tags.")
             # Let json.loads handle the error below for consistency

        if not content_to_parse:
            current_error = "Validation failed: Content became empty after stripping tags/fences/whitespace."
            logger.warning(f"parse_and_validate_structured_json: {current_error}") # Warning if content becomes empty
        else:
            # --- End Fix 2 ---
            try:
                parsed_object = json.loads(content_to_parse)
                logger.debug("Successfully parsed potentially stripped backend content as JSON.")
                # Now validate against schema
                try:
                    jsonschema.validate(instance=parsed_object, schema=json_schema)
                    logger.debug("JSON object successfully validated against the schema.")
                except ValidationError as e:
                    current_error = (
                        f"Validation failed: JSON does not match the schema. Error: {e.message} "
                        f"on instance path: {'/'.join(map(str, e.path))}."
                    )
                    logger.error(f"parse_and_validate_structured_json: {current_error}") # Error for validation failure
                    logger.error(f"--- Offending JSON Snippet ---")
                    try: logger.error(json.dumps(parsed_object, indent=2)[:500] + "...")
                    except Exception: logger.error(str(parsed_object)[:500] + "...")
                    logger.error(f"----------------------------")
                except Exception as e: # Catch other jsonschema errors
                    current_error = f"Validation failed: An unexpected error occurred during schema validation: {e}"
                    logger.exception(f"parse_and_validate_structured_json: {current_error}") # Exception for unexpected validation errors

            except json.JSONDecodeError as e:
                # Error occurred even after attempting to strip tags/fences
                error_detail = f"Error: {e}."
                if original_content_was_fenced:
                    error_detail += " Content was fenced."
                current_error = f"Validation failed: Backend content is not valid JSON after stripping. {error_detail} Content attempted: '{content_to_parse[:200]}...'"
                logger.error(f"parse_and_validate_structured_json: {current_error}") # Error for JSON decode errors

    # Update state: set error if one occurred, otherwise clear it for this step
    # Keep parsed_object if parsing happened but validation failed
    if current_error:
        # Error already logged above
        return {"error": current_error, "parsed_json_object": parsed_object}
    else:
        # Success
        return {"parsed_json_object": parsed_object, "error": None}


async def check_text_for_tool_calls(state: GraphState) -> Dict[str, Any]:
    """ Parses backend content FOR TOOL CALLS (Only run in 'text' mode). """
    logger.debug("--- Node: check_text_for_tool_calls ---")
    backend_content = state.get('backend_response_content')
    parsed_tool_calls = None
    error = None

    if backend_content is None:
        logger.warning("Backend content is None in check_text_for_tool_calls (text mode). Treating as no tool calls.") # Warning
    elif isinstance(backend_content, str):
        # Use the existing helper to detect embedded tool call JSON
        detected_tool_calls, _ = _parse_backend_response_for_tool_calls(backend_content) # Helper logs internally
        if detected_tool_calls:
            num_tools = len(detected_tool_calls)
            logger.debug(f"Parsed {num_tools} tool call(s).")
            parsed_tool_calls = detected_tool_calls
        else:
            logger.debug("Parsed 0 tool calls (Text response).")
    else:
        # Content is not None and not string - unexpected
        error = f"Parsing failed: Expected string content for text/tool mode, got {type(backend_content)}"
        logger.error(f"check_text_for_tool_calls: {error}") # Error for unexpected type

    # Update state: set error if one occurred, otherwise clear it for this step
    return {"parsed_tool_calls": parsed_tool_calls, "error": error}


async def format_structured_json_response(state: GraphState) -> Dict[str, Any]:
    """ Formats the final OpenAI response for validated JSON object output. """
    logger.debug("--- Node: format_structured_json_response ---")
    parsed_json_object = state.get('parsed_json_object')
    backend_data = state.get('backend_response_data')
    initial_request = state['initial_request']
    current_error = None

    if parsed_json_object is None:
        current_error = "Cannot format JSON response: Validated JSON object is missing."
        logger.error(f"format_structured_json_response: {current_error}") # Error
    elif backend_data is None:
         current_error = "Cannot format JSON response: Backend response data is missing."
         logger.error(f"format_structured_json_response: {current_error}") # Error

    final_response_payload = None
    if not current_error:
        try:
            backend_usage = backend_data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            backend_model_name = backend_data.get("model", initial_request.get("model", "unknown"))
            # Stringify the validated JSON for the content field
            content_string = json.dumps(parsed_json_object)

            openai_response_message = {
                "role": "assistant", "content": content_string, "tool_calls": None,
            }
            final_response_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": backend_model_name,
                "choices": [{"index": 0, "message": openai_response_message, "logprobs": None, "finish_reason": "stop"}], # finish_reason is 'stop' for JSON mode
                "usage": backend_usage,
            }
            logger.debug("Formatted JSON object response for client.")
        except Exception as e:
            current_error = f"Error formatting JSON response payload: {e}"
            logger.exception(f"format_structured_json_response: {current_error}") # Exception

    # Update state
    return {"final_openai_response": final_response_payload, "error": current_error}


async def format_tool_calls_response(state: GraphState) -> Dict[str, Any]:
    """ Formats the final OpenAI response when tool calls are detected (Text Mode). """
    logger.debug("--- Node: format_tool_calls_response ---")
    parsed_tool_calls = state.get('parsed_tool_calls')
    backend_data = state.get('backend_response_data')
    initial_request = state['initial_request']
    current_error = None
    final_response_payload = None

    if not parsed_tool_calls:
        current_error = "Cannot format tool call response: Parsed tool calls are missing."
        logger.error(f"format_tool_calls_response: {current_error}") # Error
    elif not backend_data:
        current_error = "Cannot format tool call response: Backend response data is missing."
        logger.error(f"format_tool_calls_response: {current_error}") # Error

    if not current_error:
        try:
            backend_usage = backend_data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            backend_model_name = backend_data.get("model", initial_request.get("model", "unknown"))
            tool_calls_for_openai = []
            for tool_call_info in parsed_tool_calls:
                tool_call_id = f"call_{uuid.uuid4().hex}"
                # Ensure arguments are dumped to string
                arguments_str = json.dumps(tool_call_info.get("arguments", {}))
                tool_calls_for_openai.append({
                    "id": tool_call_id, "type": "function",
                    "function": { "name": tool_call_info["name"], "arguments": arguments_str }
                })
            openai_response_message = { "role": "assistant", "content": None, "tool_calls": tool_calls_for_openai }
            final_response_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": backend_model_name,
                "choices": [{"index": 0, "message": openai_response_message, "logprobs": None, "finish_reason": "tool_calls"}],
                "usage": backend_usage,
            }
            logger.debug("Formatted tool call response for client.")
        except Exception as e:
            current_error = f"Error formatting tool call response payload: {e}"
            logger.exception(f"format_tool_calls_response: {current_error}") # Exception

    # Update state
    return {"final_openai_response": final_response_payload, "error": current_error}


async def format_plain_text_response(state: GraphState) -> Dict[str, Any]:
    """ Formats the final OpenAI response for a normal text reply (Text Mode). """
    logger.debug("--- Node: format_plain_text_response ---")
    # Use backend_response_content which *might* be None if backend returned null content
    backend_content = state.get('backend_response_content', "") # Default to empty string if None
    backend_data = state.get('backend_response_data')
    initial_request = state['initial_request']
    current_error = None
    final_response_payload = None

    if backend_data is None:
        current_error = "Cannot format text response: Backend response data is missing."
        logger.error(f"format_plain_text_response: {current_error}") # Error

    if not current_error:
        try:
            backend_usage = backend_data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            # Safely get finish_reason, default to 'stop'
            backend_finish_reason = backend_data.get("choices", [{}])[0].get("finish_reason", "stop")
            backend_model_name = backend_data.get("model", initial_request.get("model", "unknown"))

            openai_response_message = { "role": "assistant", "content": backend_content, "tool_calls": None }
            final_response_payload = {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()), "model": backend_model_name,
                "choices": [{"index": 0, "message": openai_response_message, "logprobs": None, "finish_reason": backend_finish_reason}],
                "usage": backend_usage,
            }
            logger.debug("Formatted text response for client.")
        except Exception as e:
            current_error = f"Error formatting text response payload: {e}"
            logger.exception(f"format_plain_text_response: {current_error}") # Exception

    # Update state
    return {"final_openai_response": final_response_payload, "error": current_error}


async def format_error_response(state: GraphState) -> Dict[str, Any]:
    """ Formats a generic error response payload based on the error in the state. """
    logger.debug("--- Node: format_error_response ---")
    error_msg = state.get("error", "Unknown error occurred during processing.")
    logger.error(f"Preparing final error response: {error_msg}") # Log the final error being formatted

    # Format according to OpenAI error schema
    final_error_payload = {
        "object": "error",
        "message": str(error_msg), # Ensure it's a string
        "type": "api_error", # Default type, could be refined
        "param": None,
        "code": None # Could try to map based on error content later
    }
    # Return the formatted error payload, keeping the error message in state
    # Ensure error is set, even if it was somehow cleared before this node
    return {"final_openai_response": final_error_payload, "error": error_msg or "Unknown error"}


# 3. Define Helper functions (Add logging where necessary)

def _prepare_json_mode_messages(messages: List[Dict[str, Any]], json_schema: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ Adds instructions to the system prompt specifically for JSON output matching a schema. """
    modified_messages = [msg.copy() for msg in messages]
    schema_string = "{}" # Default to empty object if schema is None
    if isinstance(json_schema, dict):
        try: schema_string = json.dumps(json_schema, indent=2)
        except Exception as e:
            logger.warning(f"Could not serialize JSON schema for prompt: {e}") # Warning
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
    for msg in modified_messages:
        if msg.get("role") == "system":
            msg["content"] = msg.get("content", "") + json_instructions
            system_message_found = True; break
    if not system_message_found:
        modified_messages.insert(0, {"role": "system", "content": json_instructions.strip()})
    # logger.debug(f"Prepared JSON mode messages: {modified_messages}") # Potentially very verbose
    return modified_messages

def _prepare_prompt_messages(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """ Prepares messages for text mode, potentially including tool usage instructions. """
    if not tools: return messages
    modified_messages = [msg.copy() for msg in messages]
    try: tool_descriptions = "\n\nAvailable Tools:\n" + json.dumps(tools, indent=2)
    except Exception as e:
        logger.warning(f"Could not serialize tools for prompt: {e}") # Warning
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
    for msg in modified_messages:
        if msg.get("role") == "system":
            msg["content"] = msg.get("content", "") + tool_descriptions + instructions
            system_message_found = True; break
    if not system_message_found:
        modified_messages.insert(0, {"role": "system", "content": (tool_descriptions + instructions).strip()})
    processed_messages = []
    for i, msg in enumerate(modified_messages):
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown_id")
            content = msg.get("content", "")
            processed_messages.append({"role": "user", "content": f"[Tool execution result for call {tool_call_id}]:\n{content}"})
        else: processed_messages.append(msg)
    # logger.debug(f"Prepared tool mode messages: {processed_messages}") # Potentially very verbose
    return processed_messages

def _parse_backend_response_for_tool_calls(backend_response_content: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """ Detects specific tool call JSON within a potentially larger text response. """
    if not isinstance(backend_response_content, str): return None, backend_response_content
    content_trimmed = backend_response_content.strip()
    match = re.search(r"(\{[\s\n]*\"tool_calls\":\s*\[.*?\][\s\n]*\})", content_trimmed, re.DOTALL)
    if not match: return None, backend_response_content
    potential_json_str = match.group(1)
    try:
        data = json.loads(potential_json_str)
        if isinstance(data, dict) and "tool_calls" in data:
            tool_calls_list = data["tool_calls"]
            if isinstance(tool_calls_list, list) and all(isinstance(call, dict) and "name" in call and "arguments" in call for call in tool_calls_list):
                if not tool_calls_list: return None, backend_response_content
                valid_calls = []
                for call in tool_calls_list:
                    args = call.get("arguments"); parsed_args = None
                    if isinstance(args, dict): parsed_args = args
                    elif isinstance(args, str):
                        try:
                            parsed_args = json.loads(args)
                            if not isinstance(parsed_args, dict):
                                logger.warning(f"Parsed arguments for tool '{call.get('name')}' is not a dict ({type(parsed_args)}). Using raw string.") # Warning
                                parsed_args = args
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse arguments string for tool '{call.get('name')}'. Using raw string.") # Warning
                            parsed_args = args
                    else:
                        logger.warning(f"Unexpected argument type ({type(args)}) for tool '{call.get('name')}'. Using empty dict.") # Warning
                        parsed_args = {}
                    valid_calls.append({"name": call["name"], "arguments": parsed_args})
                return valid_calls, backend_response_content
        return None, backend_response_content
    except json.JSONDecodeError: return None, backend_response_content


# 4. Define Conditional Edge Logic (Add logging)

def route_on_llm_status(state: GraphState) -> Literal["wait_and_retry", "route_by_format", "format_error_response"]:
    """ Routes based on the classified status after the LLM call. """
    status = state.get("llm_call_status")
    logger.debug(f"--- Decision: route_on_llm_status ---")
    logger.debug(f"LLM Call Status: {status}")

    if status == "retryable_error":
        logger.debug("Decision -> wait_and_retry")
        return "wait_and_retry"
    elif status == "success":
        logger.debug("Decision -> route_by_format")
        return "route_by_format"
    else: # Covers "fatal_error" or unexpected None status
        if status != "fatal_error":
            logger.warning(f"Unexpected llm_call_status '{status}', proceeding to error handling.") # Warning for unexpected status
        logger.debug("Decision -> format_error_response")
        return "format_error_response"

def route_on_response_format(state: GraphState) -> Literal["parse_and_validate_structured_json", "check_text_for_tool_calls"]:
    """ Routes successful responses based on the initially requested format type. """
    response_format_type = state.get("response_format_type", "text") # Default to text if missing
    logger.debug(f"--- Decision: route_on_response_format ---")
    logger.debug(f"Requested Response Format: {response_format_type}")

    if response_format_type == "json_object":
        logger.debug("Decision -> parse_and_validate_structured_json")
        return "parse_and_validate_structured_json"
    else: # 'text' mode
        logger.debug("Decision -> check_text_for_tool_calls")
        return "check_text_for_tool_calls"

def check_json_validation_result(state: GraphState) -> Literal["format_structured_json_response", "format_error_response"]:
    """ Checks if JSON parsing and validation succeeded. """
    error = state.get('error')
    parsed_json = state.get('parsed_json_object')
    logger.debug(f"--- Decision: check_json_validation_result ---")
    logger.debug(f"Error State (Post-Validation): {error}, Parsed JSON: {'Yes' if parsed_json is not None else 'No'}")

    if error or parsed_json is None:
        error_reason = error or 'No object parsed'
        logger.debug(f"Decision -> format_error_response (JSON Validation Failed: {error_reason})")
        # Ensure error state reflects the failure if it wasn't already set
        if not error: state['error'] = "JSON validation failed: No valid object was parsed."
        return "format_error_response"
    else:
        logger.debug("Decision -> format_structured_json_response")
        return "format_structured_json_response"

def route_text_mode_result(state: GraphState) -> Literal["format_tool_calls_response", "format_plain_text_response", "format_error_response"]:
    """ Determines how to format the final response in TEXT mode (tool call vs plain text). """
    error = state.get('error') # Check if error occurred *during* tool parsing
    parsed_tool_calls = state.get('parsed_tool_calls')
    logger.debug(f"--- Decision: route_text_mode_result (Text Mode) ---")
    logger.debug(f"Error State (Post-Text/Tool Parsing): {error}, Parsed Tool Calls Found: {'Yes' if parsed_tool_calls else 'No'}")

    if error: # If check_text_for_tool_calls itself failed critically
        logger.debug(f"Decision -> format_error_response (Error during text/tool parsing: {error})")
        return "format_error_response"
    elif parsed_tool_calls:
        logger.debug("Decision -> format_tool_calls_response")
        return "format_tool_calls_response"
    else:
        # No error and no tool calls found -> format as plain text
        logger.debug("Decision -> format_plain_text_response")
        return "format_plain_text_response"


# 5. Build the Graph (Refactored Structure - No changes needed here)
workflow = StateGraph(GraphState)
# ... (Add nodes as before) ...
workflow.add_node("prepare_request", prepare_request)
workflow.add_node("invoke_backend_llm", invoke_backend_llm)
workflow.add_node("check_llm_call_status", check_llm_call_status) # New node
workflow.add_node("wait_and_retry", wait_and_retry)
workflow.add_node("route_by_format", route_by_format) # New node
workflow.add_node("parse_and_validate_structured_json", parse_and_validate_structured_json)
workflow.add_node("format_structured_json_response", format_structured_json_response)
workflow.add_node("check_text_for_tool_calls", check_text_for_tool_calls)
workflow.add_node("format_tool_calls_response", format_tool_calls_response)
workflow.add_node("format_plain_text_response", format_plain_text_response)
workflow.add_node("format_error_response", format_error_response)

# ... (Define edges as before) ...
workflow.set_entry_point("prepare_request")
workflow.add_edge("prepare_request", "invoke_backend_llm")
workflow.add_edge("invoke_backend_llm", "check_llm_call_status")
workflow.add_conditional_edges(
    "check_llm_call_status",
    route_on_llm_status,
    {"wait_and_retry": "wait_and_retry", "route_by_format": "route_by_format", "format_error_response": "format_error_response"}
)
workflow.add_edge("wait_and_retry", "invoke_backend_llm")
workflow.add_conditional_edges(
    "route_by_format",
    route_on_response_format,
    {"parse_and_validate_structured_json": "parse_and_validate_structured_json", "check_text_for_tool_calls": "check_text_for_tool_calls"}
)
workflow.add_conditional_edges(
    "parse_and_validate_structured_json",
    check_json_validation_result,
    {"format_structured_json_response": "format_structured_json_response", "format_error_response": "format_error_response"}
)
workflow.add_conditional_edges(
    "check_text_for_tool_calls",
    route_text_mode_result,
    {"format_tool_calls_response": "format_tool_calls_response", "format_plain_text_response": "format_plain_text_response", "format_error_response": "format_error_response"}
)
workflow.add_edge("format_structured_json_response", END)
workflow.add_edge("format_tool_calls_response", END)
workflow.add_edge("format_plain_text_response", END)
workflow.add_edge("format_error_response", END)


# 6. Compile the Graph (Add logging)
try:
    langgraph_app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.") # Info level
except Exception as e:
    logger.critical(f"FATAL ERROR: LangGraph workflow compilation failed: {e}", exc_info=True) # Critical + traceback
    # Optionally exit or raise to prevent server start with broken graph
    raise e # Re-raise the exception to stop Uvicorn

# Optional: Visualize the graph (Add logging)
try:
    if 'langgraph_app' in locals():
        png_bytes = langgraph_app.get_graph(xray=True).draw_mermaid_png()
        graph_filename = "flow.png"
        with open(graph_filename, "wb") as f:
            f.write(png_bytes)
        logger.info(f"Saved refactored graph visualization to {graph_filename}") # Info
    else:
        logger.warning("Skipping graph visualization because compilation failed.") # Warning
except ImportError:
     logger.warning("Skipping graph visualization: `pygraphviz` not installed or `graphviz` system library missing.") # Warning
except Exception as e:
    logger.error(f"Could not draw graph: {e}", exc_info=True) # Error + traceback


# === FastAPI Endpoints ===

# --- Specific Endpoint: Chat Completions (using Enhanced LangGraph + Logging) ---
@app.post("/v1/chat/completions")
async def chat_completions_langgraph_proxy(request: Request):
    """ Handles /v1/chat/completions using the LangGraph app, supporting structured JSON output. """
    request_id = uuid.uuid4()
    logger.info(f"Received request {request_id} for /v1/chat/completions from {request.client.host}")
    try:
        openai_payload = await request.json()
        logger.debug(f"Request {request_id} Payload ---")
        try: logger.debug(json.dumps(openai_payload, indent=2))
        except Exception as e: logger.debug(f"(Error pretty-printing payload: {e}) Payload: {openai_payload}")
        logger.debug("---------------------------------")
    except json.JSONDecodeError:
        logger.error(f"Request {request_id}: Invalid JSON payload received.")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    stream = openai_payload.get("stream", False)
    if stream:
        logger.warning(f"Request {request_id}: Streaming requested, but LangGraph execution is non-streaming. Returning full response.")
        # Consider raising error: raise HTTPException(status_code=400, detail="Streaming is not supported")

    # Initial state for the graph
    initial_state: GraphState = {
        "initial_request": openai_payload,
        "response_format_type": "text", "json_schema": None, "backend_payload": None,
        "backend_response_data": None, "backend_response_content": None, "llm_call_status": None,
        "parsed_tool_calls": None, "parsed_json_object": None, "final_openai_response": None,
        "error": None, "retry_count": -1,
        "max_retries": openai_payload.get("max_retries", DEFAULT_MAX_RETRIES)
    }

    logger.info(f"Invoking LangGraph for request {request_id}...")
    start_time = time.time()
    try:
        # Invoke the LangGraph app
        final_state = await langgraph_app.ainvoke(initial_state)
        end_time = time.time()
        logger.info(f"LangGraph execution for request {request_id} finished in {end_time - start_time:.2f}s")

    except Exception as graph_exec_error:
        # Catch unexpected errors during graph execution itself
        logger.critical(f"CRITICAL: Unhandled exception during LangGraph execution for request {request_id}: {graph_exec_error}", exc_info=True)
        # Return a generic internal server error
        error_payload = { "object": "error", "message": "Internal server error during request processing.", "type": "api_error", "param": None, "code": "internal_error" }
        return JSONResponse(content=error_payload, status_code=500)

    logger.debug(f"--- LangGraph Final State (Request {request_id}) ---")
    final_error = final_state.get('error')
    final_resp_obj = final_state.get('final_openai_response')
    logger.debug(f"Final LLM Call Status: {final_state.get('llm_call_status')}")
    logger.debug(f"Final Error State: {final_error}")
    logger.debug(f"Final Response Object Type: {type(final_resp_obj)}")
    if final_resp_obj:
         logger.debug(f"--- Final Response Payload Sent to Client (Request {request_id}) ---")
         try: logger.debug(json.dumps(final_resp_obj, indent=2))
         except Exception as e: logger.debug(f"(Error pretty-printing final response: {e}) Response: {final_resp_obj}")
         logger.debug("---------------------------------------------")
    else:
         logger.debug(f"--- Final Response Payload: None (Request {request_id}) ---")
    logger.debug("---------------------------")

    # Get the final response (could be success or a formatted error)
    final_response = final_state.get("final_openai_response")

    # Handle errors: Use the formatted error payload if available
    if final_response and isinstance(final_response, dict) and final_response.get("object") == "error":
         logger.error(f"Graph finished with error state for request {request_id}, returning formatted error response.")
         # Determine status code based on error type
         status_code = 500 # Default internal error
         error_msg = str(final_state.get('error', final_response.get("message", ""))) # Get error from state or response
         if "Backend API Error" in error_msg:
             try: status_code = int(error_msg.split("(")[1].split(")")[0])
             except: status_code = 502 # Bad Gateway if parsing fails
         elif "timed out" in error_msg: status_code = 504
         elif "Could not connect" in error_msg: status_code = 503
         elif "Validation failed" in error_msg or "Invalid request" in error_msg or "schema" in error_msg.lower() or "invalid during startup" in error_msg:
             status_code = 400 # Bad request if schema/validation fails
         logger.info(f"Request {request_id} returning error - Status: {status_code}, Message: {error_msg}")
         # Return the structured error payload
         return JSONResponse(content=final_response, status_code=status_code)

    # If graph finished but no proper final response generated (success or formatted error)
    elif not final_response:
        final_error_msg = final_state.get("error", "Unknown processing error")
        logger.critical(f"CRITICAL: Graph execution finished for request {request_id} but no final response was generated.")
        logger.error(f"Final state error context: {final_error_msg}")
        logger.error(f"--- Full Final State (No Response - Request {request_id}) ---")
        try: logger.error(json.dumps(final_state, indent=2, default=str))
        except Exception as e: logger.error(f"Error logging final state: {e}")
        logger.error("--------------------------------------")
        error_payload = { "object": "error", "message": f"Internal server error: Processing failed to produce a response. Detail: {final_error_msg}", "type": "api_error", "param": None, "code": "processing_error" }
        return JSONResponse(content=error_payload, status_code=500)

    # Success case (JSON, Text, or Tool Call response)
    logger.info(f"Request {request_id} successful. Sending Final Response Payload from LangGraph to Client.")
    return JSONResponse(content=final_response)


# --- Generic Proxy Route (Catch-All - Add Basic Logging) ---
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def generic_proxy(request: Request, path: str):
    # Exclude the LangGraph-handled endpoint
    if path == "v1/chat/completions" and request.method == "POST":
        # This should ideally not be reached if the specific route is defined first,
        # but good practice to keep the check.
        logger.warning(f"Generic proxy received request for handled path /v1/chat/completions POST")
        raise HTTPException(status_code=404, detail="Not Found (Internal Routing Issue)")

    # Check if BACKEND_BASE_URL was validated
    if not BACKEND_BASE_URL:
        error_msg = "Cannot proxy request: BACKEND_API_URL was invalid during startup."
        logger.error(f"Generic proxy error for path '/{path}': {error_msg}")
        raise HTTPException(status_code=503, detail="Backend service is misconfigured")

    request_id = uuid.uuid4()
    logger.info(f"Generic proxy request {request_id}: Path=/{path}, Method={request.method} from {request.client.host}")

    if not path.startswith('/'): path = '/' + path
    target_url = urljoin(BACKEND_BASE_URL, path)
    logger.debug(f"Request {request_id} targeting backend URL: {target_url}")

    headers = {h: v for h, v in request.headers.items() if h.lower() not in ['host', 'content-length']}
    if YOUR_BACKEND_API_KEY: headers.setdefault("Authorization", f"Bearer {YOUR_BACKEND_API_KEY}")
    body = await request.body()
    backend_req = http_client.build_request( method=request.method, url=target_url, headers=headers, content=body, params=request.query_params )
    backend_resp = None
    try:
        start_time = time.time()
        backend_resp = await http_client.send(backend_req, stream=True)
        end_time = time.time()
        logger.info(f"Generic proxy request {request_id} to /{path} -> Backend Status: {backend_resp.status_code}, Time: {end_time - start_time:.2f}s")
        response_headers = {h: v for h, v in backend_resp.headers.items() if h.lower() not in ['content-encoding', 'transfer-encoding', 'connection']}

        async def stream_generator():
            try:
                async for chunk in backend_resp.aiter_bytes(): yield chunk
                logger.debug(f"Generic proxy stream finished for request {request_id} (/{path})")
            finally:
                 if backend_resp and not backend_resp.is_closed: await backend_resp.aclose()

        return StreamingResponse( stream_generator(), status_code=backend_resp.status_code, headers=response_headers, media_type=backend_resp.headers.get("content-type") )

    except httpx.TimeoutException:
        logger.error(f"Error (Timeout) proxying request {request_id} to backend for /{path}.")
        if backend_resp and not backend_resp.is_closed: await backend_resp.aclose()
        raise HTTPException(status_code=504, detail="Request to backend timed out")
    except httpx.RequestError as e:
        logger.error(f"Error (Connection) proxying request {request_id} to backend for /{path}: {e}")
        if backend_resp and not backend_resp.is_closed: await backend_resp.aclose()
        raise HTTPException(status_code=503, detail=f"Could not connect to backend: {e}")
    except Exception as e:
        logger.exception(f"Error (Unexpected) in generic proxy for request {request_id} (/{path}): {e}") # Use exception
        if backend_resp and not backend_resp.is_closed: await backend_resp.aclose()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    logger.debug("Health check endpoint called.")
    # You could add more checks here, e.g., try connecting to the backend
    return {"status": "ok"}

# --- How to Run (Updated for Logging and Docker) ---
# 1. Install dependencies: pip install fastapi "uvicorn[standard]" httpx python-dotenv langgraph langchain_core jsonschema pygraphviz pathlib
#    (pygraphviz requires graphviz system library: e.g., `sudo apt-get install graphviz` or `brew install graphviz`) - Optional for graph image
# 2. Save this code as main_logged.py
# 3. Create requirements.txt (see step 2 below)
# 4. Create Dockerfile (see step 3 below)
# 5. Build the image: docker build -t my-langgraph-app .
# 6. Run containers (see step 4 below)

if __name__ == "__main__":
    import uvicorn
    # Uvicorn will run this file using the app instance
    script_name = os.path.splitext(os.path.basename(__file__))[0] # Get 'main_logged'
    logger.info(f"--- Starting OpenAI Wrapper Server (Refactored LangGraph + File Logging) ---")
    logger.info(f"Port: {WRAPPER_PORT}")
    logger.info(f"Backend Target (Initial Config): {YOUR_BACKEND_API_URL}")
    logger.info(f"Backend Base URL (Parsed): {BACKEND_BASE_URL}")
    logger.info(f"Default Max Retries: {DEFAULT_MAX_RETRIES}")
    logger.info(f"Log Level (Console): {logging.getLevelName(LOG_LEVEL)}")
    logger.info(f"Log File Path: {LOG_FILE_PATH}")
    logger.info(f"----------------------------------------------------------------------------")

    # DO NOT use reload=True in the final command within a container
    # It's okay here for local dev testing if run directly with `python main_logged.py`
    uvicorn.run(f"{script_name}:app", host="0.0.0.0", port=WRAPPER_PORT, log_level=logging.getLevelName(LOG_LEVEL).lower(), reload=False) # Use reload=False for production/docker