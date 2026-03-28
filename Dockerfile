# ============================================================
# AI Clinical Assistant — Full Stack Dockerfile
# ============================================================

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    ffmpeg \
    libsndfile1 \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# ---------- Working directory ----------
WORKDIR /app

# ---------- Install Python dependencies ----------
COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# ---------- Copy full project ----------
COPY . .

# ---------- Setup Nginx ----------
RUN rm /etc/nginx/sites-enabled/default
COPY nginx.conf /etc/nginx/conf.d/default.conf

# ---------- Expose ports ----------
EXPOSE 80
EXPOSE 8000

# ---------- Backend working dir ----------
WORKDIR /app/backend

# ---------- Start both services ----------
CMD service nginx start && uvicorn main:app --host 0.0.0.0 --port 8000