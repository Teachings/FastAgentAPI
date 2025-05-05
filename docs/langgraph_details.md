# Langgraph Details
## Workflow Overview (`main_langgraph_enhanced.py`)

LangGraph models the request processing workflow as a stateful flowchart:

### Step-by-Step Process

#### 1. `prepare_request` (Entry Point)

* Receives incoming requests.
* Determines response type (`text` or `json_object`) based on request parameters.
* Validates JSON schema if needed; flags immediate errors.
* Prepares the backend-specific request payload.
* Initializes state (clears errors, sets retry counter to zero).

#### 2. `invoke_backend_llm`

* Sends payload to backend LLM API.
* Captures response or errors (network issues, backend failures).
* Records success or error details in the state.

#### 3. `check_llm_call_status`

* Evaluates backend invocation outcomes.
* Categorizes the outcome as:

  * `success`
  * `retryable_error` (temporary issues like timeouts)
  * `fatal_error` (permanent issues, invalid requests, exhausted retries)

#### 4. Conditional Branching (`route_on_llm_status`)

* **Retryable Error**: Routes to `wait_and_retry`.
* **Fatal Error**: Routes directly to `format_error_response`.
* **Success**: Proceeds to `route_by_format`.

#### 5. `wait_and_retry`

* Increments retry count.
* Waits (using exponential backoff).
* Clears transient errors.
* Loops back to retry the backend call (`invoke_backend_llm`).

#### 6. `route_by_format` (Post-Success)

* Decides processing path based on the original request’s format:

  * **JSON Requested**: `parse_and_validate_structured_json`
  * **Text Requested**: `check_text_for_tool_calls`

#### 7. JSON Processing Path

* **`parse_and_validate_structured_json`**: Parses and validates JSON content against the provided schema.
* **Validation Outcome (`check_json_validation_result`)**:

  * Success: `format_structured_json_response`
  * Failure: `format_error_response`
* **`format_structured_json_response`**: Formats validated JSON content into OpenAI response format.

#### 8. Text/Tool Processing Path

* **`check_text_for_tool_calls`**: Identifies `tool_calls` within the response.
* **Tool Call Outcome (`route_text_mode_result`)**:

  * Tool Calls Present: `format_tool_calls_response`
  * Plain Text: `format_plain_text_response`
  * Parsing Errors: `format_error_response`
* **`format_tool_calls_response`**: Packages tool calls for OpenAI compatibility.
* **`format_plain_text_response`**: Formats plain text responses.

#### 9. `format_error_response`

* Generates standardized OpenAI-compatible error responses based on stored error details.

#### 10. `END` (Completion)

* All workflows converge here.
* Final OpenAI-formatted response is delivered back to the initiating client via FastAPI endpoint.
