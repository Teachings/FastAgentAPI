import os
import json
import traceback
from openai import OpenAI, APIError
from termcolor import colored

# --- Configuration ---
WRAPPER_BASE_URL = os.environ.get("WRAPPER_URL", "http://localhost:8002/v1")
TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL", "llama3.2")
WRAPPER_API_KEY = os.environ.get("WRAPPER_API_KEY", "sk-1111")

# --- Tool Definitions ---
multi_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform calculations.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
        }
    }
]

# --- Test Case Function ---
def test_multi_tool_call():
    """R2: Tests the wrapper's ability to handle standard tool call requests."""
    print(colored("\n=== Test Case: R2 - Multi-Tool Call (Success) ===", 'magenta', attrs=['bold']))
    messages = [{"role": "user", "content": "What is the weather in London, UK, and what is 85 * 13?"}]
    try:
        client = OpenAI(
            base_url=WRAPPER_BASE_URL,
            api_key=WRAPPER_API_KEY,
            timeout=180.0,
            max_retries=0
        )
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=messages,
            tools=multi_tools,
            tool_choice="auto"
        )
        if response.choices and response.choices[0].message and response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            print(colored(f"SUCCESS: Received {len(tool_calls)} tool call(s).", 'green'))
            for tc in tool_calls:
                args = tc.function.arguments
                print(f"  - Tool Call ID: {colored(tc.id, 'cyan')}, Function: {colored(tc.function.name, 'yellow')}, Args: {args}")
        else:
            print(colored("WARNING: No tool calls or content found in response.", 'yellow'))
    except APIError as e:
        print(colored(f"!!! API Error: Status Code {e.status_code}", 'red', attrs=['bold']))
        print(f"Error Details: {e.message} (Type: {e.body.get('type') if e.body else 'unknown'})")
    except Exception as e:
        print(colored(f"!!! Unexpected Error: {type(e).__name__} - {e}", 'red', attrs=['bold']))
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_tool_call()