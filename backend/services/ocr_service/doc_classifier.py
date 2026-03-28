from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def classify_document(text):

    prompt = f"""
You are a medical document classifier.

Classify the document into ONE category:

1. prescription (doctor medicines)
2. lab_report (blood test / diagnostic report)
3. other

Return ONLY one word.

TEXT:
{text}
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    return res.choices[0].message.content.strip().lower()