import os
from groq import Groq
from dotenv import load_dotenv

from .retriever import retrieve
from .prompts import MEDICAL_PROMPT
from .safety import is_context_relevant, is_medically_safe
from .memory import get_chat_history, save_interaction

# Load environment variables
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "llama-3.3-70b-versatile"


# ----------------------------
# Semantic Intent Classifier
# ----------------------------
def classify_query_llm(question: str):

    classification_prompt = f"""
You are a medical query classifier.

Classify the question into ONE category:

1) reference_range → asking normal values
2) interpret_value → patient result with numbers
3) educational_test → asking about the test itself
4) condition_explanation → asking meaning of abnormal result without numbers
5) lifestyle_guidance → asking about precautions, prevention, diet, exercise, or management advice

Return ONLY the category name.
Do not explain anything.

Question: {question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip().lower()


def build_sources(docs):
    sources = []

    for d in docs:
        meta = d.metadata

        source_type = meta.get("type", "medical")
        topic = meta.get("topic", "").replace("_", " ").title()

        if source_type == "lab":
            label = f"{topic} — Lab Reference"
        elif source_type == "qa":
            label = f"{topic} — Medical Q&A"
        elif source_type == "encyclopedia":
            label = f"{topic} — Medical Encyclopedia"
        else:
            label = f"{topic} — Medical Source"

        if label not in sources and topic:
            sources.append(label)

    if not sources:
        return ""

    text = "\n\nEvidence Used:\n"
    for s in sources:
        text += f"• {s}\n"

    return text


# ----------------------------
# Main Answer Generator
# ----------------------------
def generate_answer(question: str):

    # Step 0: Risk Filter
    if not is_medically_safe(question):
        return (
            "I cannot provide guidance on this request. "
            "Please consult a qualified healthcare professional."
        )

    # Step 1: Classify intent
    intent = classify_query_llm(question)

    # Step 2: Retrieve context
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Step 3: Get conversation history
    chat_history = get_chat_history()

    # Step 4: Route based on intent
    if intent == "reference_range":
        instruction = """
Explain the normal reference range clearly.
Do NOT assume patient value.
"""

    elif intent == "interpret_value":
        instruction = """
Compare the patient's value with the normal range.

IMPORTANT RULES:
- State whether the value is high, low, or normal.
- Explain what the test measures.
- Do NOT diagnose diseases.
- Do NOT name specific conditions unless present in context.
- Do NOT give treatment advice.
- If abnormal, say it may require medical evaluation.

Keep the explanation concise and neutral.
"""

    elif intent == "condition_explanation":
        instruction = """
Explain what this abnormal condition means in simple language.

IMPORTANT:
- Only describe what the condition is.
- Do NOT list possible causes unless they are in the provided context.
- Do NOT suggest treatments.
- Keep answer concise.
"""

    elif intent == "lifestyle_guidance":
        instruction = """
Provide general precaution or lifestyle advice in simple language.

IMPORTANT:
- Give safe, general guidance only.
- Do NOT replace doctor treatment.
- Do NOT suggest stopping medications.
- Keep advice practical and clear.
"""

    elif intent == "educational_test":
        instruction = """
Explain what this medical test measures and why it is important.
"""

    else:
        instruction = """
Explain clearly in simple medical language.
"""

    # Step 5: Choose system role
    if intent in ["interpret_value", "reference_range"]:
        system_role = "You are a clinical lab report interpreter. Only explain based on the provided medical data."

    elif intent == "lifestyle_guidance":
        system_role = "You are a medical educator providing safe general health guidance."

    else:
        system_role = "You are a medical educator explaining health concepts to a patient in simple terms."

    # Step 6: Build final prompt (WITH MEMORY)
    final_prompt = f"""
    Conversation History:
    {chat_history}

    Medical Context:
    {context}

    Current Question:
    {question}

    Instruction:
    {instruction}
    """

    # Step 7: Generate final answer
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    # attach citations
    sources = build_sources(docs)
    final_answer = answer + sources

    save_interaction(question, final_answer)

    return final_answer