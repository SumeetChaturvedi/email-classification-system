import requests
import json
import time
from typing import Dict, Any

def wait_for_api(url: str, max_retries: int = 5, delay: int = 2) -> bool:
    """Wait for the API to become available."""
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            print(f"API not available yet, retrying in {delay} seconds... (Attempt {i+1}/{max_retries})")
            time.sleep(delay)
    return False

def test_classification():
    """Test the email classification endpoint."""
    # API endpoint
    url = "http://localhost:8000/api/v1/classify"
    
    # Test cases
    test_cases = [
        {
            "email_body": "Hello, my name is John Doe and my email is johndoe@example.com. I'm having trouble with my billing."
        },
        {
            "email_body": "Hi, I need to reset my password. My phone number is 123-456-7890."
        },
        {
            "email_body": "Please update my credit card ending in 1234. My CVV is 123."
        },
        {
            "email_body": "My Aadhar number is 1234-5678-9012 and I need to update my account details."
        },
        {
            "email_body": "I was born on 01/01/1990 and need to verify my identity."
        }
    ]
    
    headers = {"Content-Type": "application/json"}
    
    print("Testing email classification API...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print("-" * 50)
        
        try:
            # Make API request
            response = requests.post(url, json=test_case, headers=headers)
            response.raise_for_status()
            
            # Get response data
            result = response.json()
            
            # Print results
            print(f"Input Email: {result['input_email_body']}")
            print(f"Category: {result['category']}")
            print(f"Masked Email: {result['masked_email']}")
            print(f"Demasked Email: {result['demasked_email']}")
            print("\nMasked Entities:")
            for entity in result['masked_entities']:
                print(f"- {entity['classification']}: {entity['entity']}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
        
        print("\n")

def test_health():
    """Test the health check endpoint."""
    url = "http://localhost:8000/api/v1/health"
    
    print("Testing health check endpoint...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")

def main():
    """Main test function."""
    # Wait for API to be available
    health_url = "http://localhost:8000/api/v1/health"
    if not wait_for_api(health_url):
        print("API is not available. Please make sure the server is running.")
        return
    
    # Test health endpoint
    test_health()
    print("\n" + "="*60 + "\n")
    
    # Test classification endpoint
    test_classification()

if __name__ == "__main__":
    main()