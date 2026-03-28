# backend/orchestrator/medical_agent.py

from services.rag_service.pipeline.generator import generate_answer
from services.risk_service.predict_risk import predict_heart_risk
from services.vision_service.vision_explainer import explain_medical_image
from services.ocr_service.summarize_report import interpret_medical_report


# ------------------------
# TEXT ROUTER
# ------------------------

def handle_text_query(query: str):

    query = query.lower()

    if "risk" in query or "heart disease" in query:
        return {"service": "risk_service"}

    if "xray" in query or "scan" in query:
        return {"service": "vision_service"}

    if "fever" in query or "symptom" in query:
        answer = generate_answer(query)
        return {"service": "rag", "response": answer}

    # fallback
    answer = generate_answer(query)
    return {"service": "rag", "response": answer}


# ------------------------
# OCR REPORT HANDLER
# ------------------------

def handle_report(file_path):

    result = interpret_medical_report(file_path)

    return {
        "service": "ocr_service",
        "response": result
    }


# ------------------------
# IMAGE HANDLER
# ------------------------

def handle_xray(image_path):

    result = explain_medical_image(image_path)

    return {
        "service": "vision_service",
        "response": result
    }


# ------------------------
# RISK HANDLER
# ------------------------

def handle_risk(data):

    result = predict_heart_risk(data)

    return {
        "service": "risk_service",
        "response": result
    }