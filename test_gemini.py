import os
from google import genai

with open("key.txt") as f:
    key = f.read().strip()
print("Key found:", bool(key), "| length:", len(key))

try:
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with one word: working?",
    )
    print("Reply:", resp.text)
except Exception as e:
    print("Error:", type(e).__name__, "-", e)