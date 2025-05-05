# test_wrapper_client_all_scenarios_termcolor.py
import os
import json
import time
import traceback # Re-import standard traceback
from openai import OpenAI, APIError, APITimeoutError, InternalServerError

# --- termcolor Import ---
# Using termcolor for terminal output
# Install with: pip install termcolor
try:
    from termcolor import colored
except ImportError:
    print("Warning: 'termcolor' library not found. Output will not be colored.")
    print("Install using: pip install termcolor")
    # Define a dummy colored function if termcolor is not available
    def colored(text, *args, **kwargs):
        return text

# --- Configuration ---
WRAPPER_BASE_URL = os.environ.get("WRAPPER_URL", "http://localhost:8002/v1")
TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL", "llama3.2")
WRAPPER_API_KEY = os.environ.get("WRAPPER_API_KEY", "sk-1111")

# --- Helper Function for JSON Output (No internal coloring) ---
def pretty_print_json(data, title="JSON Data"):
    """Prints JSON data with a colored title."""
    json_string = ""
    is_error = False
    try:
        if isinstance(data, str):
             # Try to load/dump for consistent formatting
             loaded_data = json.loads(data)
             json_string = json.dumps(loaded_data, indent=2)
        elif isinstance(data, dict) or isinstance(data, list):
             json_string = json.dumps(data, indent=2)
        else:
             json_string = repr(data) # Represent other types
             title += " (Non-JSON Data)"
             is_error = True

    except (json.JSONDecodeError, TypeError) as e:
        json_string = f"Error processing data: {e}\nOriginal Data: {data}"
        title += " (Processing Error)"
        is_error = True

    title_color = 'red' if is_error else 'cyan'
    print(colored(f"--- {title} ---", title_color))
    print(json_string)
    print(colored("-" * (len(title) + 8), title_color))


# --- Print Initial Config ---
print(colored("--- OpenAI Client Test Script (All Scenarios) ---", 'magenta', attrs=['bold']))
print(f"Wrapper URL: {colored(WRAPPER_BASE_URL, 'cyan')}")
print(f"Target Model: {colored(TARGET_MODEL_NAME, 'cyan')}")


# --- Initialize OpenAI Client ---
try:
    client = OpenAI(
        base_url=WRAPPER_BASE_URL,
        api_key=WRAPPER_API_KEY,
        timeout=180.0,
        max_retries=0
    )
    print(colored("OpenAI client initialized successfully.", 'green'))
except Exception as e:
    print(colored("\n!!! Error initializing OpenAI client !!!", 'red', attrs=['bold']))
    print(e) # Standard exception printing
    print("Please ensure the WRAPPER_URL is correct and accessible.")
    exit(1)

# # --- Tool Definitions ---
# multi_tools = [
#     {"type": "function", "function": {"name": "get_current_weather", "description": "Get the current weather.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}}},
#     {"type": "function", "function": {"name": "calculator", "description": "Perform calculations.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
# ]

# --- Define Tools (same as curl example) ---
multi_tools = [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a specific city.",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "The city and country, e.g. London, UK"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}
          },
          "required": ["location"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "calculator",
        "description": "Perform basic arithmetic calculations.",
        "parameters": {
          "type": "object",
          "properties": {"expression": {"type": "string", "description": "The mathematical expression to evaluate, e.g., \"85 * 13\""}},
          "required": ["expression"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "parameters": {
          "type": "object",
          "properties": {"ticker": {"type": "string", "description": "The stock ticker symbol, e.g., AAPL"}},
          "required": ["ticker"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "unit_converter",
        "description": "Convert values between different units (e.g., miles to km, USD to EUR).",
        "parameters": {
          "type": "object",
          "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
          "required": ["value", "from_unit", "to_unit"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "simple_search",
        "description": "Perform a simple web search for a query.",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }
]

# --- Test Case Functions ---

# R1: Simple Text Response
def test_simple_text_response():
    """R1: Tests a basic request expecting a plain text response."""
    print(colored("\n--- Test Case: R1 - Text Response (Success) ---", 'magenta', attrs=['bold']))
    print(colored("*Requesting a simple text answer...*", 'blue'))
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=messages
        )
        pretty_print_json(response.model_dump(), title="Full Response Received")

        print(colored("\nValidation:", 'blue', attrs=['bold']))
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            print(colored("SUCCESS:", 'green'), "Received text response as expected.")
            print(colored("--- Content ---", 'green'))
            print(response.choices[0].message.content)
            print(colored("---------------", 'green'))
        elif response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
             print(colored("WARNING:", 'yellow'), "Received tool_calls unexpectedly.")
        else:
            print(colored("WARNING:", 'yellow'), "Response received but no text content found.")
            if response.choices: print(colored(f"Finish Reason:", 'yellow'), f"{response.choices[0].finish_reason}")

    except APIError as e:
        print(colored(f"!!! API Error:", 'red', attrs=['bold']), f"Status Code: {e.status_code}")
        error_details = e.message
        error_type = "api_error"
        if e.body and isinstance(e.body, dict):
            error_details = e.body.get('message', error_details)
            error_type = e.body.get('type', error_type)
            print(colored("Error Details:", 'red'), f"{error_details} (Type: {error_type})")
            pretty_print_json(e.body, title="Raw Error Body")
        else:
            print(colored("Error Details:", 'red'), f"{error_details}")
    except Exception as e:
        print(colored(f"!!! Unexpected Error during R1: {type(e).__name__} - {e} !!!", 'red', attrs=['bold']))
        traceback.print_exc() # Print standard traceback
    print("-" * 50)

# R2: Multi-Tool Call
def test_multi_tool_call():
    """R2: Tests the wrapper's ability to handle standard tool call requests."""
    print(colored("\n--- Test Case: R2 - Multi-Tool Call (Success) ---", 'magenta', attrs=['bold']))
    print(colored("*Requesting weather and calculation via tools...*", 'blue'))
    messages = [{"role": "user", "content": "What is the weather in London, UK, and what is 85 * 13?"}]
    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=messages,
            tools=multi_tools,
            tool_choice="auto"
        )
        pretty_print_json(response.model_dump(), title="Full Response Received")

        print(colored("\nValidation:", 'blue', attrs=['bold']))
        if response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            print(colored("SUCCESS:", 'green'), f"Received {len(tool_calls)} tool call(s).")
            for tc in tool_calls:
                 args = tc.function.arguments
                 try:
                     parsed_args = json.loads(args)
                     print(f"  - Tool Call ID: {colored(tc.id, 'cyan')}, Function: {colored(tc.function.name, 'yellow')}, Args: {json.dumps(parsed_args)}")
                 except json.JSONDecodeError:
                      print(f"  - Tool Call ID: {colored(tc.id, 'cyan')}, Function: {colored(tc.function.name, 'yellow')}, Args (raw string): {args}")

        elif response.choices and response.choices[0].message and response.choices[0].message.content is not None:
            print(colored("INFO:", 'yellow'), "Received text response instead of tool calls:")
            print(colored("--- Text Content ---", 'yellow'))
            print(response.choices[0].message.content)
            print(colored("------------------", 'yellow'))
        else:
            print(colored("WARNING:", 'yellow'), "Response received but no tool calls or content found.")
            if response.choices: print(colored("Finish Reason:", 'yellow'), f"{response.choices[0].finish_reason}")

    except APIError as e:
        print(colored(f"!!! API Error:", 'red', attrs=['bold']), f"Status Code: {e.status_code}")
        error_details = e.message
        error_type = "api_error"
        if e.body and isinstance(e.body, dict):
            error_details = e.body.get('message', error_details)
            error_type = e.body.get('type', error_type)
            print(colored("Error Details:", 'red'), f"{error_details} (Type: {error_type})")
            pretty_print_json(e.body, title="Raw Error Body")
        else:
             print(colored("Error Details:", 'red'), f"{error_details}")
    except Exception as e:
        print(colored(f"!!! Unexpected Error during R2: {type(e).__name__} - {e} !!!", 'red', attrs=['bold']))
        traceback.print_exc()
    print("-" * 50)

# R3: Retry Scenario Simulation
def test_retry_scenario_simulation():
    """R3: Simulates the *outcome* of a retry by making a standard request."""
    print(colored("\n--- Test Case: R3 - Retry Scenario (Simulated Outcome) ---", 'magenta', attrs=['bold']))
    print(colored("*Simulating a request that would succeed after a hypothetical retry.*", 'blue'))
    print(colored("Note:", 'yellow'), "This test doesn't force a retry, just shows a successful call.")
    messages = [{"role": "user", "content": "Tell me about the Grand Canyon."}]
    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=messages
        )
        pretty_print_json(response.model_dump(), title="Full Response Received (after potential wrapper retry)")

        print(colored("\nValidation:", 'blue', attrs=['bold']))
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            print(colored("SUCCESS:", 'green'), "Received text response (simulating successful outcome after retry).")
            print(colored("--- Content ---", 'green'))
            print(response.choices[0].message.content)
            print(colored("---------------", 'green'))
        else:
            print(colored("WARNING:", 'yellow'), "No text content found in response.")
            if response.choices: print(colored("Finish Reason:", 'yellow'), f"{response.choices[0].finish_reason}")

    except APIError as e:
        print(colored(f"!!! API Error during simulated retry outcome test:", 'red', attrs=['bold']), f"Status Code: {e.status_code}")
        error_details = e.message
        error_type = "api_error"
        if e.body and isinstance(e.body, dict):
            error_details = e.body.get('message', error_details)
            error_type = e.body.get('type', error_type)
            print(colored("Error Details:", 'red'), f"{error_details} (Type: {error_type})")
            pretty_print_json(e.body, title="Raw Error Body")
        else:
            print(colored("Error Details:", 'red'), f"{error_details}")
    except Exception as e:
        print(colored(f"!!! Unexpected Error during R3: {type(e).__name__} - {e} !!!", 'red', attrs=['bold']))
        traceback.print_exc()
    print("-" * 50)


# R4: JSON Mode with Schema (Success)
def test_json_mode_with_schema():
    """R4: Tests the wrapper's ability to handle JSON mode requests WITH a schema."""
    print(colored("\n--- Test Case: R4 - JSON Mode Request with Schema (Success) ---", 'magenta', attrs=['bold']))
    print(colored("*Requesting user info extraction into a specific JSON schema...*", 'blue'))
    user_info_schema = {
      "type": "object",
      "properties": {
        "name": {"type": "string", "description": "The person's full name."},
        "occupation": {"type": "string", "description": "The person's occupation or job title."},
        "city": {"type": "string", "description": "The city where the person lives."}
      },
      "required": ["name", "occupation", "city"]
    }
    messages = [
        {"role": "system", "content": "Extract information from the user text."},
        {"role": "user", "content": "The patient's name is Sarah Connor, she works as a programmer, and resides in Los Angeles."}
    ]
    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=messages,
            response_format={ "type": "json_object", "schema": user_info_schema }
        )
        pretty_print_json(response.model_dump(), title="Full Response Received")

        print(colored("\nValidation:", 'blue', attrs=['bold']))
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content_raw = response.choices[0].message.content
            print(f"Received raw content (expecting stringified JSON):")
            print(colored("--- Raw Content ---", 'grey'))
            print(content_raw)
            print(colored("-------------------", 'grey'))
            try:
                parsed_json = json.loads(content_raw)
                print(colored("\nSUCCESS:", 'green'), "Response content is valid JSON.")
                pretty_print_json(parsed_json, title="Parsed JSON Object")
                # Basic check for required keys (client-side sanity check)
                if all(key in parsed_json for key in user_info_schema.get("required", [])):
                     print(colored("SUCCESS:", 'green'), "Parsed JSON contains required keys.")
                else:
                     print(colored("WARNING:", 'yellow'), "Parsed JSON might be missing required keys (client-side check).")

            except json.JSONDecodeError as json_err:
                print(colored("\nERROR:", 'red', attrs=['bold']), f"Response content is NOT valid JSON: {json_err}")
                print(f"Content attempted for parsing: '{content_raw}'")
        elif response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
             print(colored("\nERROR:", 'red', attrs=['bold']), "Received tool_calls unexpectedly when JSON mode with schema was requested.")
        else:
            print(colored("\nWARNING:", 'yellow'), "Response received but no message content found.")
            if response.choices: print(colored("Finish Reason:", 'yellow'), f"{response.choices[0].finish_reason}")

    except APIError as e:
        print(colored(f"!!! API Error:", 'red', attrs=['bold']), f"Status Code: {e.status_code}")
        error_details = e.message
        error_type = "api_error"
        if e.body and isinstance(e.body, dict):
            error_details = e.body.get('message', error_details)
            error_type = e.body.get('type', error_type)
            print(colored("Error Details:", 'red'), f"{error_details} (Type: {error_type})")
            pretty_print_json(e.body, title="Raw Error Body")
        else:
             print(colored("Error Details:", 'red'), f"{error_details}")
    except Exception as e:
        print(colored(f"!!! Unexpected Error during R4: {type(e).__name__} - {e} !!!", 'red', attrs=['bold']))
        traceback.print_exc()
    print("-" * 50)


# R5: JSON Schema Validation Error Attempt
def test_json_schema_validation_error_attempt():
    """R5: Attempts to trigger a schema validation error from the wrapper."""
    print(colored("\n--- Test Case: R5 - JSON Schema Validation Error (Attempt) ---", 'magenta', attrs=['bold']))
    print(colored("*Requesting JSON with a schema, hoping the LLM fails to comply perfectly (e.g., age as string)...*", 'blue'))
    print(colored("Note:", 'yellow'), "Success/failure depends on the backend LLM and wrapper validation.")

    simple_schema = {
      "type": "object",
      "properties": { "name": {"type": "string"}, "age": {"type": "number"} },
      "required": ["name", "age"]
    }
    messages = [{"role": "user", "content": "Create a JSON object for a user named Alex who is 'twenty five' years old."}]

    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=messages,
            response_format={ "type": "json_object", "schema": simple_schema }
        )
        pretty_print_json(response.model_dump(), title="Full Response Received (Expected potential error)")

        print(colored("\nValidation:", 'blue', attrs=['bold']))
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            content_raw = response.choices[0].message.content
            print(f"Received raw content:")
            print(colored("--- Raw Content ---", 'grey'))
            print(content_raw)
            print(colored("-------------------", 'grey'))
            try:
                parsed_json = json.loads(content_raw)
                print(colored("\nWARNING:", 'yellow'), "Received valid JSON. Wrapper might not have detected schema mismatch, or LLM complied.")
                pretty_print_json(parsed_json, title="Parsed JSON Object")
                # Check type client-side
                if isinstance(parsed_json.get("age"), (int, float)):
                     print(colored("Client Check:", 'green'), "Age is a number.")
                else:
                     print(colored("Client Check:", 'red', attrs=['bold']), f"Age is NOT a number (type: {type(parsed_json.get('age')).__name__}). Schema validation likely failed if wrapper checked types.")

            except json.JSONDecodeError as json_err:
                print(colored("\nERROR:", 'red', attrs=['bold']), f"Response content was NOT valid JSON: {json_err}")
        else:
             print(colored("\nWARNING:", 'yellow'), "No message content found.")
             if response.choices: print(colored("Finish Reason:", 'yellow'), f"{response.choices[0].finish_reason}")

    except APIError as e:
        print(colored(f"!!! API Error:", 'red', attrs=['bold']), f"Status Code: {e.status_code}")
        error_details = e.message
        error_type = "api_error"
        if e.body and isinstance(e.body, dict):
            error_details = e.body.get('message', error_details)
            error_type = e.body.get('type', error_type)
            # Check if it's the expected validation error type from the wrapper
            if error_type == 'llm_output_validation_error':
                 print(colored("SUCCESS (Expected Error):", 'green', attrs=['bold']), "Received schema validation error from wrapper.")
            else:
                 print(colored("WARNING:", 'yellow'), "Received an API error, but not the specific validation error type.")

            print(colored("Error Details:", 'red'), f"{error_details} (Type: {error_type})")
            pretty_print_json(e.body, title="Raw Error Body")
        else:
             print(colored("Error Details:", 'red'), f"{error_details}")
    except Exception as e:
        print(colored(f"!!! Unexpected Error during R5: {type(e).__name__} - {e} !!!", 'red', attrs=['bold']))
        traceback.print_exc()
    print("-" * 50)

# R6: Fatal Error Simulation Placeholder
def test_fatal_error_simulation_placeholder():
    """R6: Placeholder explaining that fatal errors cannot be forced by the client."""
    print(colored("\n--- Test Case: R6 - Fatal Error Scenario (Placeholder) ---", 'magenta', attrs=['bold']))
    print(colored("Note:", 'yellow'), "Cannot reliably force a fatal backend error (e.g., 503 Service Unavailable) from the client.")
    print(colored("This scenario would typically be triggered by actual backend issues or reaching max retries.", 'blue'))
    print(colored("If such an error occurred, the APIError exception would be caught, similar to the R5 error handling.", 'blue'))
    # Example of how the catch block would look:
    print(colored("\n--- Example Error Output (if a 503 occurred) ---", 'red'))
    print(colored("!!! API Error:", 'red', attrs=['bold']), "Status Code: 503")
    print(colored("Error Details:", 'red'), "Backend LLM request failed. Status Code: 503. Response: Service Unavailable (Type: backend_llm_error)")
    print(colored("--- Raw Error Body ---", 'red'))
    print("""{
  "error": {
    "message": "Backend LLM request failed. Status Code: 503. Response: Service Unavailable",
    "type": "backend_llm_error",
    "param": null,
    "code": "service_unavailable"
  }
}""")
    print(colored("------------------------------------------------", 'red'))
    print("-" * 50)


# --- Main Execution ---
if __name__ == "__main__":
    print(colored("\n🚀 Welcome to the OpenAI Client Test Script Demo!", 'magenta', attrs=['bold']))
    print(colored("Let's walk through each test case step-by-step. Press Enter to proceed.", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n--- Step 1: Initialize the OpenAI Client ---", 'green', attrs=['bold']))
    print(colored("This will connect to the wrapper server. Let's check for any errors.", 'yellow', attrs=['bold']))
    print(colored("✅ Press Enter to start the initialization...", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    # Initialize client
    try:
        client = OpenAI(
            base_url=WRAPPER_BASE_URL,
            api_key=WRAPPER_API_KEY,
            timeout=180.0,
            max_retries=0
        )
        print(colored("✅ OpenAI client initialized successfully.", 'green', attrs=['bold']))
    except Exception as e:
        print(colored("❌ Error initializing client. Check the WRAPPER_URL and API key.", 'red', attrs=['bold']))
        print(colored(f"Error details: {e}", 'red', attrs=['bold']))
        print(colored("🚨 Press Enter to exit...", 'red', attrs=['bold']))
        input()
        exit(1)
    print(colored("✅ Initialization successful. Ready for the next step.", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n--- Step 2: Test Case R1 - Simple Text Response ---", 'cyan', attrs=['bold']))
    print(colored("We'll request a plain text answer. Let's see the output.", 'yellow', attrs=['bold']))
    print(colored("🔁 Press Enter to run R1...", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter
    test_simple_text_response()
    print(colored("✅ R1 completed. Ready for R2.", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n--- Step 3: Test Case R2 - Multi-Tool Call ---", 'cyan', attrs=['bold']))
    print(colored("This test will trigger multiple function calls (e.g., weather + calculator).", 'yellow', attrs=['bold']))
    print(colored("🔁 Press Enter to run R2...", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter
    test_multi_tool_call()
    print(colored("✅ R2 completed. Ready for R3.", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n--- Step 4: Test Case R3 - Retry Simulation ---", 'cyan', attrs=['bold']))
    print(colored("Simulating a retry scenario. Let's observe the output.", 'yellow', attrs=['bold']))
    print(colored("🔁 Press Enter to run R3...", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter
    test_retry_scenario_simulation()
    print(colored("✅ R3 completed. Ready for R4.", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n--- Step 5: Test Case R4 - JSON Mode with Schema ---", 'cyan', attrs=['bold']))
    print(colored("This tests JSON output validation. Let's see if the schema is enforced.", 'yellow', attrs=['bold']))
    print(colored("🔁 Press Enter to run R4...", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter
    test_json_mode_with_schema()
    print(colored("✅ R4 completed. Ready for R5.", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n--- Step 6: Test Case R5 - Schema Validation Error ---", 'cyan', attrs=['bold']))
    print(colored("We'll attempt to trigger a schema validation error. Watch for the response.", 'yellow', attrs=['bold']))
    print(colored("🔁 Press Enter to run R5...", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter
    test_json_schema_validation_error_attempt()
    print(colored("✅ R5 completed. Ready for R6.", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n--- Step 7: Test Case R6 - Fatal Error Placeholder ---", 'cyan', attrs=['bold']))
    print(colored("This section explains how fatal errors are handled. No actual error is triggered.", 'yellow', attrs=['bold']))
    print(colored("🔁 Press Enter to run R6...", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter
    test_fatal_error_simulation_placeholder()
    print(colored("✅ R6 completed. All tests done!", 'cyan', attrs=['bold']))
    input()  # Wait for user to press Enter

    print(colored("\n🎉 All tests completed! Let's review the results.", 'magenta', attrs=['bold']))
    print(colored("Thank you for watching! Let me know if you have questions.", 'cyan', attrs=['bold']))