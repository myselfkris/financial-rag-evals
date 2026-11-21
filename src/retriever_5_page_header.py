"""
Experiment H1b: does adding the page's own top (table name, units, years) help?

ONE change vs retriever_4_header (current best, H1):
    the header also includes the first PAGE_TOP_CHARS characters of the page,
    with whitespace collapsed. For a financial table that is usually:
    "3M Company and Subsidiaries Consolidated Statement of Cash Flows
     Years ended December 31 (Millions) 2018 2017 2016 ..."

Everything else is identical to H1: same company/filing header, same
chunk_text() (1000 chars, 200 overlap), same embedding model, same top_k.

Detail: chunk 0 already starts with the page top, so it gets only the H1
header. Repeating the page top there would waste the embedding's 256-token budget.
"""

import json
import re
import chromadb

from src.retriever_3 import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP
from src.retriever_4_header import make_header

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financebench_h1b_page_header"
PAGE_TOP_CHARS = 200


def page_top(text: str) -> str:
    """First PAGE_TOP_CHARS characters of the page, with whitespace collapsed."""
    return re.sub(r"\s+", " ", text).strip()[:PAGE_TOP_CHARS]


def build_index():
    client = chromadb.PersistentClient(path=DB_PATH)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME)

    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    ids, documents, metadatas = [], [], []
    for doc in docs:
        base_header = make_header(doc)                       # same as H1
        full_header = f"{base_header}\n{page_top(doc['text'])} ..."  # <-- the only change

        for idx, chunk in enumerate(chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            header = base_header if idx == 0 else full_header
            ids.append(f"{doc['doc_id']}_chunk_{idx}")
            documents.append(f"{header}\n{chunk}")
            metadatas.append({
                "doc_id": doc["doc_id"],
                "company": doc["company"],
                "doc_name": doc["doc_name"],
                "page_num": str(doc.get("page_num", "N/A")),
                "chunk_index": idx,
            })

    print(f"⏳ Indexing {len(documents)} chunks with company + page-top header into '{COLLECTION_NAME}'...")
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )
    print("✅ H1b page-header index built!")


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
