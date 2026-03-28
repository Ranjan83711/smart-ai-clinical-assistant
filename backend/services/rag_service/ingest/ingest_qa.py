import os
import pandas as pd
import re
from rich import print
from rich.progress import track

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import DATA_PATH, VECTOR_DB_PATH, EMBED_MODEL

CSV_FILE = os.path.join(DATA_PATH, "QA medical dataset.csv")


# ---------------------------
# Text Cleaning
# ---------------------------
def clean_text(text: str) -> str:
    text = str(text)

    # remove html tags
    text = re.sub(r"<.*?>", " ", text)

    # remove citations [1], (1)
    text = re.sub(r"\[\d+\]|\(\d+\)", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------
# Topic Extraction (simple heuristic)
# ---------------------------
KEYWORDS = [
    "diabetes", "anemia", "hypertension", "blood pressure",
    "kidney", "creatinine", "hemoglobin", "cholesterol",
    "thyroid", "infection", "fever", "cancer", "heart"
]

def detect_topic(question: str):
    q = question.lower()
    for k in KEYWORDS:
        if k in q:
            return k.replace(" ", "_")
    return "general_medical"


# ---------------------------
# Load CSV → Documents
# ---------------------------
def load_qa_documents():
    df = pd.read_csv(CSV_FILE)

    documents = []

    for _, row in track(df.iterrows(), total=len(df), description="Processing QA pairs"):
        question = clean_text(row["question"])
        answer = clean_text(row["answer"])

        content = f"Question: {question}\nAnswer: {answer}"

        metadata = {
            "type": "qa",
            "topic": detect_topic(question),
            "focus_area": clean_text(row.get("focus_area", "general")),
            "source": clean_text(row.get("source", "medical_qa_dataset")),
            "safety_level": "medium"
        }

        documents.append(Document(page_content=content, metadata=metadata))

    return documents


# ---------------------------
# Build Vector DB
# ---------------------------
def build_vectordb(docs):

    print("[yellow]Loading embedding model...[/yellow]")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[yellow]Creating / Updating Chroma DB...[/yellow]")

    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    BATCH_SIZE = 64   # very important

    print(f"[cyan]Embedding {len(docs)} documents in batches...[/cyan]")

    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        db.add_documents(batch)
        print(f"[green]Processed {i+len(batch)}/{len(docs)}[/green]")

    db.persist()

    print(f"[bold green]Stored {len(docs)} QA medical knowledge chunks[/bold green]")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    print("[bold cyan]Starting QA ingestion pipeline...[/bold cyan]")

    documents = load_qa_documents()
    build_vectordb(documents)

    print("[bold green]QA knowledge base ready![/bold green]")