import os
import torch
from PIL import Image
from torchvision import transforms
from groq import Groq
from dotenv import load_dotenv

from .training.model import get_model

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Paths
# -------------------------
CHEST_MODEL_PATH = os.path.join("services/vision_service/weights/chest_xray_resnet18.pth")
FRACTURE_MODEL_PATH = os.path.join("services/vision_service/weights/fracture_resnet18.pth")

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -------------------------
# Load models once
# -------------------------
def load_models():
    chest_model = get_model(2)
    chest_model.load_state_dict(torch.load(CHEST_MODEL_PATH, map_location=DEVICE))
    chest_model.eval().to(DEVICE)

    fracture_model = get_model(2)
    fracture_model.load_state_dict(torch.load(FRACTURE_MODEL_PATH, map_location=DEVICE))
    fracture_model.eval().to(DEVICE)

    return chest_model, fracture_model


CHEST_MODEL, FRACTURE_MODEL = load_models()

# -------------------------
# Predict
# -------------------------
def predict_image(image_path):

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        chest_out = CHEST_MODEL(img)
        fracture_out = FRACTURE_MODEL(img)

    chest_prob = torch.softmax(chest_out, dim=1)[0]
    fracture_prob = torch.softmax(fracture_out, dim=1)[0]

    pneumonia_conf = chest_prob[1].item()
    fracture_conf = fracture_prob[0].item()

    if pneumonia_conf > fracture_conf:
        return {
            "type": "chest_xray",
            "finding": "pneumonia" if pneumonia_conf > 0.5 else "normal",
            "confidence": round(pneumonia_conf, 2)
        }
    else:
        return {
            "type": "bone_xray",
            "finding": "fracture" if fracture_conf > 0.5 else "normal",
            "confidence": round(fracture_conf, 2)
        }

# -------------------------
# Structured Clinical Finding
# -------------------------
def build_radiology_finding(pred):

    conf = pred["confidence"]

    # Bone X-ray
    if pred["type"] == "bone_xray":

        if pred["finding"] == "fracture":
            if conf >= 0.90:
                certainty = "findings are consistent with a fracture"
            elif conf >= 0.75:
                certainty = "findings are highly suggestive of a fracture"
            else:
                certainty = "findings are suspicious for a fracture"

            return f"""
Radiographic Finding:
There is cortical disruption in the imaged bone; {certainty}.
Model confidence: {conf:.2f}
No additional osseous abnormalities identified.
"""
        else:
            return f"""
Radiographic Finding:
No definite radiographic evidence of fracture.
Model confidence: {conf:.2f}
"""

    # Chest X-ray
    if pred["type"] == "chest_xray":

        if pred["finding"] == "pneumonia":
            if conf >= 0.90:
                certainty = "findings are consistent with pneumonia"
            elif conf >= 0.75:
                certainty = "findings are highly suggestive of pneumonia"
            else:
                certainty = "findings are suspicious for pneumonia"

            return f"""
Radiographic Finding:
There are patchy lung opacities; {certainty}.
Model confidence: {conf:.2f}
"""
        else:
            return f"""
Radiographic Finding:
No acute cardiopulmonary abnormality detected.
Model confidence: {conf:.2f}
"""

# -------------------------
# Clinical Radiology Report
# -------------------------
def generate_clinical_report(finding_text):

    prompt = f"""
You are an AI clinical radiology assistant.

Convert the following finding into a short professional radiology report.

STRICT RULES:
- Use probabilistic medical language (consistent with / suggestive of / suspicious for)
- Never claim definitive diagnosis
- No disease education
- Maximum 3 sentences
- Formal medical tone
- End with: Clinical correlation is recommended.

Finding:
{finding_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.05
    )

    return response.choices[0].message.content

# -------------------------
# Patient Friendly Explanation
# -------------------------
def simplify_for_patient(clinical_text):

    prompt = f"""
You are a doctor explaining an imaging result to a patient in a calm and clear way.

Rewrite the report in very simple language.

Rules:
- Only explain what is seen in the scan
- Do NOT mention symptoms, pain, or how the patient feels
- Do NOT sound certain (use may/might/appears)
- Avoid medical terms
- Keep it supportive and easy to understand
- 3–4 short lines only
- End with: Please visit a doctor to confirm and guide treatment.

Medical Report:
{clinical_text}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35
    )

    return response.choices[0].message.content
# -------------------------
# Final Pipeline
# -------------------------
def explain_medical_image(image_path):

    prediction = predict_image(image_path)
    finding_text = build_radiology_finding(prediction)

    clinical_report = generate_clinical_report(finding_text)
    patient_explanation = simplify_for_patient(clinical_report)

    return {
        "prediction": prediction,
        "clinical_report": clinical_report,
        "patient_explanation": patient_explanation
    }