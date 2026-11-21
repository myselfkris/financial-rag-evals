import json
import chromadb

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financebench_basic_1000"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into fixed-size character chunks with overlap, avoiding tiny trailing duplicates."""
    if not text:
        return []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def build_index():
    client = chromadb.PersistentClient(path=DB_PATH)

    # Delete only this baseline collection if it exists (preserves hybrid collections)
    existing_names = [col.name for col in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(name=COLLECTION_NAME)

    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    ids, documents, metadatas = [], [], []

    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc['doc_id']}_chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "doc_id": doc["doc_id"],
                "company": doc["company"],
                "doc_name": doc["doc_name"],
                "page_num": str(doc.get("page_num", "N/A")),
                "chunk_index": idx
            })

    print(f"⏳ Indexing {len(documents)} chunks ({CHUNK_SIZE} chars) into ChromaDB...")

    # Batch inserts to avoid SQLite parameter limits on larger corpora
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )

    print("✅ Basic 1000-char Index built!")


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "doc_id": meta["doc_id"],
            "doc_name": meta["doc_name"],
            "page_num": meta.get("page_num", "N/A"),
            "text": results["documents"][0][i]
        })

    return retrieved


if __name__ == "__main__":
    build_index()