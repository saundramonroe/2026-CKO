import requests
import json

API_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE4MDQzNTA3MzQsImtpZCI6IjYiLCJzY29wZXMiOlsiY2xvdWQ6cmVhZCIsImNsb3VkOndyaXRlIiwicmVwbzpyZWFkIiwicmVwbzp3cml0ZSJdLCJ2ZXIiOiJhcGk6MSIsInN1YiI6IjVjZjQwNDJhLTE3NDUtNDE0MC1iODE5LTA4NmRjZGE1Njc2NiJ9.AitpSHHtfS1aWrDgfBrvYVpdvbm8_jb66uv95tB-cegtpAiRuMsDTCsNJ5PkBYW65E0ZF5kLfE_3UOxDr-EekEuJa2Z4Pxix7CJ0e1rmo4WHx-8dKcEX5lh5VvI5AUHpTfHO_ovZK3wB6edcz4ROtACVhAPbgaEyCoeJuKSkHGTpGGgV-fb7irWEDlrLTGyYEuuRMJoJlQsD6QPlaFnGcOl_c1WFUTyoZaGFHlDbYm2Aef1-GPHDXgZzjH3xHvwAHLghawJPmHYMlJM9awKoKStTlUqOp-5i_f0S_RkrQY6_m9jNDNeogVuoszyNUgfFNz8FyRvGBJ-z4EqjgHrQqA"

CONNECT_ENDPOINT = "https://ahadji1.sb.anacondaconnect.com/api/ai/inference/serve/72b797bd-964e-4c90-9081-c64028ba383a/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

payload = {
    "messages": [
        {"role": "system", "content": "You are a fraud scorer. Return only a decimal number between 0.0 and 1.0. No explanation."},
        {"role": "user", "content": "Merchant: BITCOIN ATM, Amount: $2500. Fraud risk score:"}
    ],
    "max_tokens": 10,
    "temperature": 0.1,
    "stop": ["\n"]
}

try:
    response = requests.post(CONNECT_ENDPOINT, headers=headers, json=payload, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Raw response: {response.text[:500]}")

    if response.status_code == 200:
        data = response.json()
        # Try both response formats
        choice = data["choices"][0]
        text = choice.get("text") or choice.get("message", {}).get("content", "")
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        print(f"\nPrompt tokens: {prompt_tokens}  (should be > 20)")
        print(f"Model output: '{text}'")

except Exception as e:
    print(f"Error: {e}")