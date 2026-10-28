# CliniqAI — Smart AI Clinical Assistant

Built at **Hackfest 2026 (Naresh i Technologies)**, March 2026.

**Live demo:** https://smart-ai-clinical-assistant-3cx2.onrender.com/

Healthcare help is often delayed: there aren't enough doctors, medical reports are hard to read, and patients have nowhere to ask a quick question. CliniqAI is a single web app where a patient can:

- **Upload a lab report or prescription** and get it explained in plain language, with abnormal values highlighted
- **Upload a chest X-ray or bone X-ray** and get a screening result for pneumonia or fracture, with a simple explanation
- **Check heart-disease risk** from basic clinical values (age, BP, cholesterol, etc.)
- **Ask medical questions** by text, answered from a medical knowledge base (RAG)
- **Talk to it by voice** and hear the answer spoken back

> ⚠️ **Not a medical device.** This is a hackathon prototype for education and triage support. It does not diagnose or replace a doctor.

---

## Features

| Feature | Endpoint | How it works |
|---|---|---|
| Report analysis (OCR) | `POST /analyze_report` | EasyOCR reads the image → an LLM classifies it as a *prescription* or *lab report* → for lab reports, each test value is compared against the reference range printed on the report and flagged **LOW**, **HIGH** or normal → Llama 3.3 70B writes a patient-friendly summary |
| X-ray screening (CV) | `POST /analyze_xray` | Two fine-tuned **ResNet-18** models (PyTorch) predict **pneumonia** (chest X-ray) and **fracture** (bone X-ray) with a confidence score (threshold 0.5); the finding is turned into a radiology-style note and explained by the LLM |
| Heart-disease risk (ML) | `POST /predict_risk` | **Logistic Regression** (scikit-learn) trained on the **UCI Cleveland heart-disease dataset** (13 features, StandardScaler). Probability is bucketed into **Low (<15%) / Borderline (<40%) / Elevated (<70%) / High**, and the LLM explains the result in simple words |
| Medical Q&A (RAG) | `POST /ask` | Question → medical-safety check (Llama 3.1 8B) → retrieval from a **ChromaDB** vector store (`all-MiniLM-L6-v2` embeddings, top-k with token trimming) → answer from **Llama 3.3 70B** with sources, using the recent chat history for follow-up questions |
| Voice assistant | `POST /voice_query` | **Whisper** speech-to-text → GPT-4o-mini answer → **gpt-4o-mini-tts** text-to-speech; returns the transcription, the answer text and an MP3 of the spoken reply |

All features are available from one frontend (`frontend/index.html`).

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
 Browser ──────▶│ Nginx :80                                     │
 (index.html)   │   /       → static frontend                   │
                │   /api/*  → FastAPI :8000                     │
                └──────────────────────┬───────────────────────┘
                                       │
                         backend/main.py  (FastAPI)
                                       │
                      orchestrator/medical_agent.py
          ┌──────────────┬─────────────┼──────────────┬──────────────┐
          ▼              ▼             ▼              ▼              ▼
     ocr_service    vision_service  risk_service   rag_service   voice_service
     EasyOCR +      ResNet-18 x2    LogReg +       ChromaDB +    Whisper +
     Llama 3.3      + Llama 3.3     Llama 3.3      Llama 3.3     GPT-4o-mini + TTS
```

LLM calls for OCR, vision, risk and RAG go through **Groq** (`llama-3.3-70b-versatile`, plus `llama-3.1-8b-instant` for the safety checks). The voice service uses the **OpenAI API**.

---

## Tech stack

- **Backend:** Python 3.10, FastAPI, Uvicorn, Pydantic
- **LLMs:** Groq (Llama 3.3 70B, Llama 3.1 8B), OpenAI (GPT-4o-mini, Whisper, gpt-4o-mini-tts)
- **RAG:** LangChain, ChromaDB, sentence-transformers (`all-MiniLM-L6-v2`)
- **OCR:** EasyOCR (PaddleOCR and OpenCV also installed)
- **Computer vision:** PyTorch, torchvision (ResNet-18)
- **ML:** scikit-learn (Logistic Regression), pandas, NumPy
- **Frontend:** single-page HTML/CSS/JS
- **Deployment:** Docker (Nginx + Uvicorn in one container) on Render

---

## Project structure

```
backend/
  main.py                      FastAPI app and all endpoints
  orchestrator/
    medical_agent.py           routes each request to the right service
  services/
    ocr_service/               ocr_reader.py (EasyOCR), doc_classifier.py,
                               lab_analyzer.py (normal/abnormal ranges),
                               summarize_report.py (patient summary)
    vision_service/            vision_explainer.py (ResNet-18 pneumonia + fracture)
    risk_service/              train_model.py, predict_risk.py,
                               model.pkl, scaler.pkl, processed_cleveland.csv
    rag_service/
      ingest/                  ingest_qa.py (medical Q&A CSV),
                               ingest_encyclopedia.py (medical encyclopedia PDF),
                               ingest_lab.py (lab-report PDFs)
      pipeline/                retriever.py, generator.py, safety.py, memory.py
    voice_service/             speech_to_text.py, llm_handler.py,
                               text_to_speech.py, voice_router.py
frontend/
  index.html                   CliniqAI web app
Dockerfile                     builds backend + Nginx in one image
nginx.conf                     serves the frontend, proxies /api to FastAPI
requirements.txt
```

---

## Run locally

### 1. Environment variables

Create a `.env` file in `backend/`:

```env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key      # only needed for the voice assistant
```

### 2. Install and start the backend

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### 3. Open the frontend

Open `frontend/index.html` in a browser, or run the full stack with Docker:

```bash
docker build -t cliniqai .
docker run -p 80:80 --env-file backend/.env cliniqai
```

Then open http://localhost.

### Files not included in this repo

Some large or third-party files are not committed, so they need to be added before every feature works:

| Needed by | Path | How to get it |
|---|---|---|
| X-ray screening | `backend/services/vision_service/weights/chest_xray_resnet18.pth`, `fracture_resnet18.pth` and `vision_service/training/model.py` | Train ResNet-18 on a chest X-ray (pneumonia) dataset and a bone fracture dataset, or copy the trained weights |
| Medical Q&A | `backend/services/rag_service/data/` (Q&A CSV, encyclopedia PDF, lab-report PDFs) | Add the source files, then run the ingest scripts below to build `rag_service/vectordb/` |

```bash
cd backend
python -m services.rag_service.ingest.ingest_qa
python -m services.rag_service.ingest.ingest_encyclopedia
python -m services.rag_service.ingest.ingest_lab
```

Retrain the heart-risk model (optional; `model.pkl` is already included):

```bash
cd backend/services/risk_service
python train_model.py
```

---

## API examples

```bash
# Ask a medical question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the common symptoms of dengue?"}'

# Analyze a lab report image
curl -X POST http://localhost:8000/analyze_report \
  -F "file=@backend/services/ocr_service/General-Check-Up-Report.webp"

# Heart-disease risk
curl -X POST http://localhost:8000/predict_risk \
  -H "Content-Type: application/json" \
  -d '{"age":54,"sex":1,"cp":0,"trestbps":130,"chol":246,"fbs":0,"restecg":1,"thalach":150,"exang":0,"oldpeak":1.0,"slope":1,"ca":0,"thal":2}'
```

---

## Limitations and next steps

- The X-ray models only cover pneumonia and fractures, and the risk model only covers heart disease.
- Normal/abnormal detection depends on the reference ranges being readable in the OCR output.
- Chat history is kept in memory, so it resets when the server restarts.
- Next: add the trained weights through Git LFS or a download script, add automated tests, and support more report types and languages (Hindi).

---

## Team

- **Ranjan Kumar Yadav** — [GitHub](https://github.com/Ranjan83711) · [LinkedIn](https://www.linkedin.com/in/ranjan-yadav-05b62a231/)
- **Gautam Kumar** — [GitHub](https://github.com/gautam625)
<!-- Add any other hackathon teammates here -->
