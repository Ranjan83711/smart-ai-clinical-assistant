from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------
# REQUEST MODELS
# ------------------------------------------------

class AskInput(BaseModel):
    query: str


class RiskInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# orchestrator
from orchestrator.medical_agent import (
    handle_text_query,
    handle_report,
    handle_xray,
    handle_risk
)

# voice assistant
from services.voice_service.voice_router import handle_voice_query

app = FastAPI(title="AI Clinical Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------

@app.get("/")
def home():
    return {"message": "AI Clinical Assistant Backend Running"}

# ------------------------------------------------
# VOICE ASSISTANT
# ------------------------------------------------

@app.post("/voice_query")
async def voice_query(audio_file: UploadFile = File(...)):

    temp_path = f"temp_{audio_file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    result = handle_voice_query(temp_path)

    os.remove(temp_path)

    return result


# ------------------------------------------------
# MEDICAL QUESTION (RAG)
# ------------------------------------------------

@app.post("/ask")
def ask_question(body: AskInput):

    result = handle_text_query(body.query)

    return result


# ------------------------------------------------
# OCR REPORT ANALYSIS
# ------------------------------------------------

@app.post("/analyze_report")
async def analyze_report(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = handle_report(temp_path)

    os.remove(temp_path)

    return result


# ------------------------------------------------
# XRAY / IMAGE ANALYSIS
# ------------------------------------------------

@app.post("/analyze_xray")
async def analyze_xray(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = handle_xray(temp_path)

    os.remove(temp_path)

    return result


# ------------------------------------------------
# RISK PREDICTION
# ------------------------------------------------

@app.post("/predict_risk")
def risk_prediction(data: RiskInput):

    result = handle_risk(data.model_dump())

    return result