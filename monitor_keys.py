import os
import requests
from dotenv import load_dotenv

def monitor_groq_key(index, api_key):
    print(f"Checking Groq Key {index + 1} ({api_key[:8]}...)...")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        if response.status_code == 200:
            headers_dict = response.headers
            print(f"  Status: ACTIVE")
            print(f"  - Remaining Requests: {headers_dict.get('x-ratelimit-remaining-requests', 'N/A')}/{headers_dict.get('x-ratelimit-limit-requests', 'N/A')}")
            print(f"  - Remaining Tokens:   {headers_dict.get('x-ratelimit-remaining-tokens', 'N/A')}/{headers_dict.get('x-ratelimit-limit-tokens', 'N/A')}")
            print(f"  - Reset Requests:      {headers_dict.get('x-ratelimit-reset-requests', 'N/A')}")
            print(f"  - Reset Tokens:        {headers_dict.get('x-ratelimit-reset-tokens', 'N/A')}")
        else:
            print(f"  Status: ERROR (Code {response.status_code})")
            print(f"  - Message: {response.text}")
    except Exception as e:
        print(f"  Status: FAILED TO CONNECT ({e})")
    print("-" * 50)

def monitor_google_key(index, api_key):
    print(f"Checking Google Key {index + 1} ({api_key[:8]}...)...")
    # Call the list models endpoint to verify key status
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"  Status: ACTIVE (Key is valid and active)")
        else:
            try:
                err_msg = response.json().get("error", {}).get("message", response.text)
            except Exception:
                err_msg = response.text
            print(f"  Status: ERROR (Code {response.status_code})")
            print(f"  - Message: {err_msg}")
    except Exception as e:
        print(f"  Status: FAILED TO CONNECT ({e})")
    print("-" * 50)

def main():
    load_dotenv()
    
    groq_keys = [v.strip() for k, v in os.environ.items() if "GROQ_API_KEY" in k and v.strip()]
    google_keys = [v.strip() for k, v in os.environ.items() if "GOOGLE_API_KEY" in k and v.strip()]
    
    print("=" * 60)
    print("           API KEY USAGE & STATUS MONITOR")
    print("=" * 60)
    
    print(f"Found {len(groq_keys)} Groq API Keys and {len(google_keys)} Google API Keys in environment.\n")
    
    if groq_keys:
        print("--- Groq API Keys ---")
        for i, key in enumerate(groq_keys):
            monitor_groq_key(i, key)
            
    if google_keys:
        print("--- Google API Keys ---")
        for i, key in enumerate(google_keys):
            monitor_google_key(i, key)

if __name__ == "__main__":
    main()
