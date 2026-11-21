import json
import chromadb

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financebench_naive_v1"

def chunk_text_naive(text: str, chunk_size: int = 500) -> list[str]:
    """Naive V1 chunker: blindly slices text every 500 characters."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def build_index():
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Delete existing collection if re-running so we start fresh
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    ids = []
    documents = []
    metadatas = []

    for doc in docs:
        chunks = chunk_text_naive(doc["text"], chunk_size=500)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{doc['doc_id']}_chunk_{idx}")
            documents.append(chunk)
            # Notice we store doc_id in metadata so our Eval can check if we pulled the right page!
            metadatas.append({
                "doc_id": doc["doc_id"],
                "company": doc["company"],
                "doc_name": doc["doc_name"]
            })

    print(f"⏳ Indexing {len(documents)} naive chunks from {len(docs)} pages into ChromaDB...")
    # Add in batches of 500
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    print("✅ Naive V1 Index built successfully!")

def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """Searches ChromaDB and returns top_k chunks with their metadata."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "doc_name": results["metadatas"][0][i]["doc_name"],
            "text": results["documents"][0][i]
        })
    return retrieved

if __name__ == "__main__":
    build_index()