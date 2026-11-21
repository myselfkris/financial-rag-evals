"""
Experiment H2: is the embedding model the bottleneck?

ONE change vs retriever_5_page_header (H1b):
    embedding model all-MiniLM-L6-v2 (Chroma default, 256 tokens)
                 -> BAAI/bge-base-en-v1.5 (512 tokens, trained for search)

Chunks are built exactly as in H1b: same company/filing header, same page-top
header, same chunk_text() (1000 chars, 200 overlap), same top_k.

BGE rule (from its model card): QUESTIONS get an instruction prefix,
PASSAGES (chunks) don't. Getting this wrong silently lowers scores.

Embeddings are normalized, so Chroma's default L2 distance ranks results the
same way cosine similarity would.
"""

import json
import chromadb

from src.retriever_3 import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP
from src.retriever_4_header import make_header
from src.retriever_5_page_header import page_top

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financebench_h2_bge_base"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    """Load the model once and reuse it (loading takes seconds; per-query reload would wreck latency)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True, batch_size=32,
                              show_progress_bar=True).tolist()


def embed_query(question: str) -> list[float]:
    return get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()


def build_chunks(docs: list[dict]):
    """Identical chunk construction to retriever_5_page_header (H1b)."""
    ids, documents, metadatas = [], [], []
    for doc in docs:
        base_header = make_header(doc)
        full_header = f"{base_header}\n{page_top(doc['text'])} ..."
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
    return ids, documents, metadatas


def build_index():
    client = chromadb.PersistentClient(path=DB_PATH)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(name=COLLECTION_NAME, metadata={"embedding_model": MODEL_NAME})

    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    ids, documents, metadatas = build_chunks(docs)

    print(f"⏳ Embedding {len(documents)} chunks with {MODEL_NAME} (first run downloads ~440 MB)...")
    embeddings = embed_passages(documents)  # <-- the only change: we embed, not Chroma's default

    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
        )
    print(f"✅ H2 index built into '{COLLECTION_NAME}'!")


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[embed_query(question)],
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
