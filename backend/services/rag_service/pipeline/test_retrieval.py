from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from ..config import VECTOR_DB_PATH, EMBED_MODEL

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

db = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=embeddings
)

def search(query):
    results = db.similarity_search(query, k=5)

    print("\n==============================")
    print("QUERY:", query)
    print("==============================\n")

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(r.page_content[:400])
        print("META:", r.metadata)


if __name__ == "__main__":
    search("What does low hemoglobin mean")
    search("normal blood pressure range")
    search("symptoms of diabetes")