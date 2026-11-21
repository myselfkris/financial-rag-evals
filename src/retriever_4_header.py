"""
Experiment H1: do chunks fail because they lack company context?

ONE change vs retriever_3 (baseline):
    every chunk gets a header "Company: <company> | Filing: <doc_name>" prepended
    before it is embedded.

Everything else is identical: same chunk_text() (1000 chars, 200 overlap),
same embedding model (Chroma default, all-MiniLM-L6-v2), same top_k.

Note: the header uses document metadata known at INDEX time (which filing we
ingested). That is not leakage. Filtering by the test case's doc_name at query
time would be leakage.
"""

import json
import chromadb

from src.retriever_3 import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financebench_h1_header"


def make_header(doc: dict) -> str:
    return f"Company: {doc['company']} | Filing: {doc['doc_name']}"


def build_index():
    client = chromadb.PersistentClient(path=DB_PATH)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    ids, documents, metadatas = [], [], []
    for doc in docs:
        header = make_header(doc)
        for idx, chunk in enumerate(chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            ids.append(f"{doc['doc_id']}_chunk_{idx}")
            documents.append(f"{header}\n{chunk}")  # <-- the only change
            metadatas.append({
                "doc_id": doc["doc_id"],
                "company": doc["company"],
                "doc_name": doc["doc_name"],
                "page_num": str(doc.get("page_num", "N/A")),
                "chunk_index": idx,
            })

    print(f"⏳ Indexing {len(documents)} chunks with company header into '{COLLECTION_NAME}'...")
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
    print("✅ H1 header index built!")


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "doc_id": meta["doc_id"],
            "doc_name": meta["doc_name"],
            "page_num": meta.get("page_num", "N/A"),
            "text": results["documents"][0][i],
        })
    return retrieved


if __name__ == "__main__":
    build_index()
