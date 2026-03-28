import os
import re
from rich import print
from rich.progress import track

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import DATA_PATH, VECTOR_DB_PATH, EMBED_MODEL

LAB_FOLDER = os.path.join(DATA_PATH, "lab_reports")


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_lab_sections(text):
    """
    Extract lab test rows from reference tables.
    We detect: Test name followed by numeric range.
    """

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = []

    buffer = []

    for line in lines:

        # start of new test row (contains units or numbers)
        if re.search(r'\d+\.?\d*\s*(mg|g|ml|µ|/|%|mmol|IU|U/L)', line, re.I):
            buffer.append(line)
            sections.append(" ".join(buffer))
            buffer = []
        else:
            buffer.append(line)

    return [clean_text(s) for s in sections if len(s) > 25]

def extract_documents():

    documents = []

    for file in os.listdir(LAB_FOLDER):
        if not file.endswith(".pdf"):
            continue

        path = os.path.join(LAB_FOLDER, file)
        print(f"[cyan]Reading {file}[/cyan]")

        loader = PyPDFLoader(path)
        pages = loader.load()

        full_text = "\n".join(p.page_content for p in pages)

        sections = split_lab_sections(full_text)

        for sec in track(sections, description="Processing lab sections"):

            test_text = f"""
                Laboratory Test: {sec}

                This is a laboratory reference range entry.
                The above values represent normal reference ranges.
                If a patient's value is above the upper limit, it may be considered high.
                If below the lower limit, it may be considered low.
            """

            documents.append(Document(
                page_content=clean_text(test_text),
                metadata={
                    "type": "lab",
                    "source": file,
                    "safety_level": "high"
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

    print(f"[yellow]Embedding {len(docs)} lab knowledge units...[/yellow]")

    for i in range(0, len(docs), 32):
        db.add_documents(docs[i:i+32])
        print(f"[green]Processed {i+32}/{len(docs)}[/green]")

    db.persist()
    print("[bold green]Lab reference knowledge added![/bold green]")


if __name__ == "__main__":
    ingest()