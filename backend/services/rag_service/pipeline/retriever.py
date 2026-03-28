from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
# from langchain_chroma import Chroma
from ..config import VECTOR_DB_PATH, EMBED_MODEL
import re

def is_lab_query(query: str):
    lab_keywords = [
    # generic numeric intent
    "range", "level", "value", "count", "ratio",
    "high", "low", "normal", "increase", "decrease",
    "elevated", "reduced", "abnormal",

    # CBC
    "hemoglobin", "hb", "platelet", "platelets", "wbc", "rbc",
    "hematocrit", "hct", "mcv", "mch", "mchc", "rdw",
    "neutrophil", "lymphocyte", "monocyte", "eosinophil", "basophil",

    # Kidney (KFT/RFT)
    "creatinine", "urea", "bun", "uric acid", "egfr",

    # Liver (LFT)
    "bilirubin", "alt", "ast", "alp", "sgpt", "sgot",
    "albumin", "globulin", "protein",

    # Diabetes
    "glucose", "sugar", "hba1c", "fasting", "pp", "postprandial",

    # Lipid profile
    "cholesterol", "triglyceride", "hdl", "ldl", "vldl",

    # Electrolytes
    "sodium", "potassium", "chloride", "bicarbonate", "calcium", "phosphorus", "magnesium",

    # Thyroid
    "tsh", "t3", "t4", "ft3", "ft4",

    # Vitamins & hormones
    "vitamin d", "vitamin b12", "ferritin", "iron",

    # urine tests
    "proteinuria", "ketone", "ph", "specific gravity"
]

    q = query.lower()
    return any(word in q for word in lab_keywords) or bool(re.search(r'\d+(\.\d+)?', q))

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

MAX_TOTAL_TOKENS = 260
MAX_CHUNK_TOKENS = 140
TOP_K = 4


def trim_to_tokens(text, max_tokens):
    tokens = tokenizer.encode(text)
    tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)


def detect_query_type(question: str):

    q = question.lower()

    # numeric lab interpretation
    if any(char.isdigit() for char in q):
        return "lab_value"

    # asking normal range
    if any(word in q for word in ["normal range", "normal value", "range"]):
        return "lab_range"

    # meaning / disease explanation
    if any(word in q for word in ["meaning", "what is", "define", "causes", "symptoms"]):
        return "explanation"

    return "general"


def retrieve(question: str, k=6):

    query_type = detect_query_type(question)

    docs = db.similarity_search_with_score(question, k=15)

    reranked = []

    for doc, score in docs:

        source = doc.metadata.get("type")

        # SOURCE PRIORITY RULES
        if query_type == "lab_value":
            # Strongly prioritize lab data
            if source == "lab":
                weight = 3.0
            elif source == "encyclopedia":
                weight = 1.2
            else:
                weight = 0.8
        elif query_type == "lab_range":
            weight = 2.0 if source == "lab" else 1.2

        elif query_type == "explanation":
            weight = 2.5 if source == "encyclopedia" else 1.0

        else:
            weight = 1.0

        adjusted_score = score * weight
        reranked.append((doc, adjusted_score))

    reranked.sort(key=lambda x: x[1])

    return [doc for doc, _ in reranked[:k]]