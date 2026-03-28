import os

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectordb")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"