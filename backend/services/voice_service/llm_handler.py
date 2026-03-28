from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def generate_response(user_query):

    prompt = f"""
You are an AI medical assistant.

Answer clearly for a patient.

Rules:
- Use simple language
- If the question is medical, explain clearly
- Answer in the same language as the user (Hindi or English)

Patient question:
{user_query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful medical AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content