import requests
import json

# Test the API
url = "http://localhost:8000/classify_email"
payload = {
    "email_body": "Hello, my name is John Doe, and my email is johndoe@example.com. I'm having trouble with my billing."
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(json.dumps(response.json(), indent=2))