import os;
os.environ["GROQ_API_KEY"] = "Groq_api_key"
import requests

base_url = "https://6006-154-192-5-123.ngrok-free.app"

payload = {
    "model": "Qwen/Qwen3.5-2B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one line"}
    ],
    "temperature": 0.2,
    "max_tokens": 100
}

try:
    response = requests.post(
        base_url + "/v1/chat/completions",
        json=payload,
        timeout=60
    )

    print("STATUS:", response.status_code)
    print("RESPONSE:")
    print(response.text)

except Exception as e:
    print("ERROR:", e)