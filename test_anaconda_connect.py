# test_endpoints.py
import requests
import json

API_TOKEN = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3OTkzNDY3MjMsImtpZCI6IjMxIiwic2NvcGVzIjpbImNsb3VkOnJlYWQiLCJjbG91ZDp3cml0ZSIsInJlcG86cmVhZCIsInJlcG86d3JpdGUiXSwic3ViIjoiNzU5ZTgyNjUtZDlhZi00ZDRjLWI4NDQtYzZmMjk1Y2M2MzY4IiwidmVyIjoiYXBpOjEifQ.lKRLL1GYU3iKpATF4wXoT08a0L_3NT6OofL4P72xbOIG_9vMXWhsoi5RN40UsACn2wQZcJA_tMsf0JEfZiu7l2ACkO-Z28xSkeqCfPi8q15ZY6M0tYFMBrezRKdBa2Lhd1ULG6wGaJdBQZqnJnnhUa-oIV-cY43B0Y9x7EJoZKTPgDaPcmGNo0UJ-FM7INsiCxZUssUkY_9_92eh2O4m57_aJ190FNXnr11s6nbTynmnmKz3hK67WKjblpk-BVhKjnL9pCcyXkfWDVXviwNxlJwVCXHX7t-HbK1orXueLS-Vd91vZzgurhj6zoi9Ww_pJ0mi9_tmLyhHSXfFymO-Iw"

BASE_URL = "https://demo.sb.anacondaconnect.com/api/ai/inference/serve/424c8a7d-5d52-4afb-bc60-81c23e83713"

# Try different endpoint formats
endpoints_to_try = [
    f"{BASE_URL}/predict",
    f"{BASE_URL}/v1/predict",
    f"{BASE_URL}/inference",
    f"{BASE_URL}/v1/inference",
    f"{BASE_URL}",
    f"{BASE_URL}/score",
]

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_TOKEN}'
}

test_payload = {
    "data": [[0] * 30],
    "merchant_description": ["TEST"],
    "amount": [100.0]
}

print("Testing different endpoint formats...\n")
print("=" * 80)

for endpoint in endpoints_to_try:
    print(f"\nTrying: {endpoint}")
    print("-" * 80)
    
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=test_payload,
            timeout=10
        )
        
        content_type = response.headers.get('content-type', '')
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {content_type}")
        
        if response.status_code == 200:
            if 'json' in content_type:
                try:
                    data = response.json()
                    print("✅ SUCCESS! This endpoint works!")
                    print(f"Response: {json.dumps(data, indent=2)[:500]}")
                    print(f"\n🎯 USE THIS ENDPOINT: {endpoint}")
                    break
                except:
                    print(f"Response (first 200 chars): {response.text[:200]}")
            else:
                print(f"Response (first 200 chars): {response.text[:200]}")
        else:
            print(f"Error: {response.reason}")
            print(f"Response (first 200 chars): {response.text[:200]}")
            
    except requests.exceptions.Timeout:
        print("❌ Timeout (>10s)")
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 80)