from google import genai
from google.genai import types
import os

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="한국의 수도는 어디야?",
    config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=200
    )
)

print(response.text)