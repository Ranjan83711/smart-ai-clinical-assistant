import joblib
import numpy as np
import os
from groq import Groq
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load model
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# Feature order (VERY IMPORTANT)
FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

# ---------------------------
# Risk level mapping
# ---------------------------
def risk_category(prob):
    if prob < 0.15:
        return "Low"
    elif prob < 0.40:
        return "Borderline"
    elif prob < 0.70:
        return "Elevated"
    else:
        return "High"

# ---------------------------
# Patient explanation
# ---------------------------
def explain_risk(prob, category):

    prompt = f"""
You are a doctor explaining heart disease risk to a normal person.

Risk probability: {prob:.2f}
Risk level: {category}

Explain in simple language.

- For Borderline risk, describe it as "slightly above ideal range"
- Avoid using the phrase "moderate chance"
- Clarify that this is a risk estimate, not a diagnosis

Rules:
- Do not use emotional reassurance (avoid phrases like "don't worry")
- Do not guarantee safety
- Use neutral and supportive tone
- Encourage lifestyle improvement
- If risk is Borderline or higher, suggest routine medical checkup
- 3–4 short lines
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content

# ---------------------------
# Main prediction function
# ---------------------------
def predict_heart_risk(patient_data: dict):
    values = pd.DataFrame([patient_data])[FEATURES]
    values_scaled = scaler.transform(values)
    prob = model.predict_proba(values_scaled)[0][1]

    category = risk_category(prob)
    explanation = explain_risk(prob, category)

    return {
        "risk_probability": round(float(prob),3),
        "risk_level": category,
        "explanation": explanation
    }