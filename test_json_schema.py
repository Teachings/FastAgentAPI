from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import json
import traceback
from openai import OpenAI, APIError
from termcolor import colored

WRAPPER_BASE_URL = os.environ.get("WRAPPER_URL", "http://localhost:8002/v1")
TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL", "your-backend-model-name")
WRAPPER_API_KEY = os.environ.get("WRAPPER_API_KEY", "dummy-key-for-wrapper")

class UserInfoSchema(BaseModel):
    """Schema for extracting user information from text"""
    name: str = Field(..., description="The person's full name.")
    occupation: str = Field(..., description="The person's occupation or job title.")
    city: str = Field(..., description="The city where the person lives.")

def validate_environment_vars():
    """Validate required environment variables"""
    if not WRAPPER_BASE_URL:
        raise ValueError("WRAPPER_URL environment variable is not set")
    if not TARGET_MODEL_NAME:
        raise ValueError("TARGET_MODEL environment variable is not set")

def run_test_case(client: OpenAI, test_case: Dict[str, str]):
    """
    Run a test case for JSON mode request with schema
    Args:
        client: OpenAI client instance
        test_case: Dictionary containing 'user_message' and 'expected_output'
    """
    print(colored(f"\n=== Test Case: {test_case['user_message']} ===", 'magenta', attrs=['bold']))
    messages = [
        {"role": "system", "content": "Extract information from the user text and structure it according to the provided JSON schema."},
        {"role": "user", "content": test_case['user_message']}
    ]
    
    try:
        response = client.chat.completions.create(
            model=TARGET_MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object", "schema": UserInfoSchema.model_json_schema()}
        )
        print(colored("\nFull Response Received:", 'cyan'))
        print(response.model_dump_json(indent=2))
        
        if response.choices[0].message.content:
            # Use built-in JSON handling instead of manual stripping
            parsed_data = response.choices[0].message.content
            parsed_dict = json.loads(parsed_data)
            validated_data = UserInfoSchema.model_validate(parsed_dict)
            print(colored("\nSUCCESS: Valid JSON matching schema:", 'green'))
            print(validated_data.model_dump_json(indent=2))
            
    except APIError as e:
        print(colored(f"!!! API Error: {e.status_code} - {e.message}", 'red', attrs=['bold']))
    except Exception as e:
        print(colored(f"!!! Unexpected Error: {type(e).__name__} - {e}", 'red', attrs=['bold']))
        traceback.print_exc()

def test_json_mode_with_schema():
    """Main test function with multiple test cases"""
    validate_environment_vars()
    
    # Create client instance
    client = OpenAI(
        base_url=WRAPPER_BASE_URL,
        api_key=WRAPPER_API_KEY,
        timeout=180.0,
        max_retries=1
    )
    
    test_cases = [
        {
            "user_message": "The patient's name is Sarah Connor, she works as a programmer, and resides in Los Angeles.",
            "expected_output": {
                "name": "Sarah Connor",
                "occupation": "programmer",
                "city": "Los Angeles"
            }
        },
        {
            "user_message": "John Doe is a software engineer living in New York City.",
            "expected_output": {
                "name": "John Doe",
                "occupation": "software engineer",
                "city": "New York City"
            }
        },
        {
            "user_message": "Invalid input without proper information",
            "expected_output": {
                "name": "",
                "occupation": "",
                "city": ""
            }
        }
    ]
    
    for test_case in test_cases:
        run_test_case(client, test_case)

if __name__ == "__main__":
    test_json_mode_with_schema()