import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or "AQ.Ab8RN6JZVGGbRQ5z6J1JAchvM3iNyMlq072CgYB39QH_GhyWbQ"

client = genai.Client(api_key=api_key)

print("Available models:")
for model in client.models.list():
    print(model.name)