'''

import requests
import json

API_URL = "http://127.0.0.1:5000/chat"

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    payload = {"question": user_input}
    headers = {"Content-Type": "application/json"}

    print(f"📤 Sending: {json.dumps(payload)}")

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response_data = response.json()
        print(f"🤖 Cooper Bot: {response_data['response']}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
    except json.JSONDecodeError:
        print("❌ Failed to decode response JSON.")
'''


import requests
import json

API_URL = "http://127.0.0.1:5000/chat"

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    payload = {"question": user_input}
    headers = {"Content-Type": "application/json"}

    print(f"📤 Sending: {json.dumps(payload)}")

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response_data = response.json()
        print(f"🤖 Cooper Bot: {response_data['response']}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
    except json.JSONDecodeError:
        print("❌ Failed to decode response JSON.")
