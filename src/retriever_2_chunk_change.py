import json
import chromadb

DB_PATH = "data/chroma_db"
COLLECTION_NAME = "financebench_naive_v1"

def chunk_text(text: str, header: str, chunk_size: int = 3000) -> list[str]:
    """V2: Large 3000-char chunks (keeps full financial tables intact) + Company Header."""
    return [f"{header}\n{text[i:i + chunk_size]}" for i in range(0, len(text), chunk_size)]

def build_index():
    client = chromadb.PersistentClient(path=DB_PATH)
    
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
        header = f"Company: {doc['company']} | Document: {doc['doc_name']} | Page: {doc.get('page_num', 'N/A')}"
        chunks = chunk_text(doc["text"], header=header, chunk_size=3000)
        for idx, chunk in enumerate(chunks):
            ids.append(f"{doc['doc_id']}_chunk_{idx}")
            documents.append(chunk)
            metadatas.append({
                "doc_id": doc["doc_id"],
                "company": doc["company"],
                "doc_name": doc["doc_name"],
                "page_num": str(doc.get("page_num", "N/A"))
            })

    print(f"⏳ Indexing {len(documents)} large V2 chunks (3000 chars + header) into ChromaDB...")
    batch_size = 500
    for i in range(0, len(documents), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size]
        )
    print("✅ V2 Index built successfully!")

def retrieve(question: str, top_k: int = 3) -> list[dict]:
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
            "page_num": results["metadatas"][0][i].get("page_num", "N/A"),
            "text": results["documents"][0][i]
        })
    return retrieved

if __name__ == "__main__":
    build_index()