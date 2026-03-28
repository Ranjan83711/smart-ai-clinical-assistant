from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SAFETY_MODEL = "llama-3.1-8b-instant"


def is_medically_safe(question: str) -> bool:
    """
    Blocks harmful medical or self-harm related queries
    """

    prompt = f"""
You are a medical safety filter.

Determine if the user question is SAFE to answer.

Block if it includes:
- self harm
- suicide
- poisoning
- replacing prescribed treatment
- dangerous home remedies
- illegal drug advice
- bypassing doctor instructions

Answer ONLY SAFE or UNSAFE.

Question: {question}
"""

    response = client.chat.completions.create(
        model=SAFETY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip().upper() == "SAFE"


def is_context_relevant(question: str, context: str) -> bool:
    """
    Checks knowledge relevance (your existing guardrail)
    """

    prompt = f"""
Check if the medical context contains enough information to answer the question safely.

Answer ONLY YES or NO.

Question: {question}

Context:
{context}
"""

    response = client.chat.completions.create(
        model=SAFETY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip().upper() == "YES"