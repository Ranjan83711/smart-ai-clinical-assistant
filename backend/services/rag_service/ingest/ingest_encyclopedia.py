import os
import re
from rich import print
from rich.progress import track

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import DATA_PATH, VECTOR_DB_PATH, EMBED_MODEL

BOOK_FILE = os.path.join(DATA_PATH, "Gale Encyclopedia of Medicine Vol. 1 (A-B).pdf")


# ---------------------------
# Detect section headings
# ---------------------------
SECTION_HEADERS = [
    "Description", "Symptoms", "Diagnosis",
    "Treatment", "Risks", "Normal results",
    "Definition", "Causes", "Prevention",
    "Prognosis", "Key Terms"
]


def split_into_topics(text):
    """
    Split encyclopedia into topic-based chunks
    """

    topics = re.split(r'\n([A-Z][A-Za-z\- ]{3,40})\n', text)

    chunks = []
    current_topic = None
    buffer = ""

    for part in topics:

        part = part.strip()

        if len(part) < 3:
            continue

        # New topic detected
        if part.istitle() and len(part.split()) <= 4:
            if current_topic and buffer:
                chunks.append((current_topic, buffer))
            current_topic = part
            buffer = ""

        else:
            buffer += "\n" + part

    if current_topic and buffer:
        chunks.append((current_topic, buffer))

    return chunks


def structure_topic(topic, content):

    structured = f"Medical Topic: {topic}\n\n"

    sections = re.split(r'\n(' + '|'.join(SECTION_HEADERS) + r')\n', content)

    for i in range(1, len(sections), 2):
        section = sections[i]
        text = sections[i+1]
        structured += f"{section}: {text}\n\n"

    return structured


def extract_documents():

    print("[cyan]Loading encyclopedia...[/cyan]")
    loader = PyPDFLoader(BOOK_FILE)
    pages = loader.load()

    full_text = "\n".join(p.page_content for p in pages)

    print("[cyan]Detecting medical topics...[/cyan]")
    topics = split_into_topics(full_text)

    documents = []

    for topic, content in track(topics, description="Structuring topics"):

        structured_text = structure_topic(topic, content)

        if len(structured_text) < 200:
            continue

        documents.append(Document(
            page_content=structured_text,
            metadata={
                "type": "encyclopedia",
                "topic": topic.lower(),
                "source": "gale_medical_book",
                "safety_level": "medium"
            }
        ))

    return documents


def ingest():

    docs = extract_documents()

    print("[yellow]Loading embedding model...[/yellow]")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    print(f"[yellow]Embedding {len(docs)} medical topics...[/yellow]")

    for i in range(0, len(docs), 20):
        db.add_documents(docs[i:i+20])
        print(f"[green]Processed {i+20}/{len(docs)}[/green]")

    db.persist()
    print("[bold green]Encyclopedia knowledge added![/bold green]")


if __name__ == "__main__":
    ingest()