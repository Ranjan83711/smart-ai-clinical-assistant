import os
from groq import Groq
from dotenv import load_dotenv

from .ocr_reader import read_text
from .doc_classifier import classify_document
from .lab_analyzer import analyze_lab_report

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def interpret_medical_report(image_path):

    # Step 1 — OCR
    raw_text = read_text(image_path)

    # Step 2 — Document Classification (NEW)
    doc_type = classify_document(raw_text)

    # If lab report, stop prescription interpretation
    if "lab" in doc_type:
        analysis = analyze_lab_report(raw_text)
        return {
            "raw_ocr": raw_text,
            "explanation": "📄 Document Type Detected: LAB REPORT\n\n" + analysis
        }

    if "other" in doc_type:
        return {
            "raw_ocr": raw_text,
            "explanation": (
                "📄 Document Type Detected: OTHER DOCUMENT\n\n"
                "The uploaded image does not appear to be a prescription."
            )
        }

    # Step 3 — Prescription Interpretation
    prompt = f"""
You are a medical assistant AI.

Convert the following lab report into a structured, user-friendly summary.

STRICT RULES:
- Do NOT write long paragraphs
- Use clear sections and bullet points
- Keep language simple for non-technical users
- Do NOT say "everything is perfect"
- Do NOT assume missing values
- If something is unclear, say "Not clearly detected"
- Some OCR text may be noisy, interpret carefully

FORMAT:

🧾 Lab Report Summary:
- Hospital:
- Test Type:

✅ Overall Status:
(one line summary)

🔬 Key Results:
- Parameter: Value → Status (Normal/High/Low) + Range

💡 What This Means:
- point 1
- point 2

🩺 Recommendation:
- point 1
- point 2

LAB REPORT:
{raw_text}
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return {
        "raw_ocr": raw_text,
        "explanation": response.choices[0].message.content
    }