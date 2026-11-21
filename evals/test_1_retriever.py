import json
import sys
from pathlib import Path

# Allow importing from src/
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.retriever import retrieve

def score_single_retrieval(retrieved_chunks: list[dict], gold_doc_ids: list[str]) -> dict:
    retrieved_doc_ids = [c["doc_id"] for c in retrieved_chunks]
    gold_set = set(gold_doc_ids)

    # 1. Coverage / Recall @ K: How many required gold docs were found at least once?
    found_gold_docs = gold_set.intersection(set(retrieved_doc_ids))
    recall = len(found_gold_docs) / len(gold_set) if gold_set else 0.0

    # 2. Rank Score (MRR): 1 / rank of the FIRST relevant chunk
    mrr = 0.0
    for rank_idx, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in gold_set:
            mrr = 1.0 / (rank_idx + 1)
            break

    # 3. Noise / Precision @ K: What fraction of retrieved chunks came from a gold doc?
    useful_chunks = sum(1 for doc_id in retrieved_doc_ids if doc_id in gold_set)
    precision = useful_chunks / len(retrieved_chunks) if retrieved_chunks else 0.0

    return {
        "recall": recall,
        "mrr": mrr,
        "precision": precision,
        "retrieved_docs": [c["doc_name"] for c in retrieved_chunks]
    }

def run_retriever_eval(top_k: int = 3):
    with open("data/dev_20.json", "r", encoding="utf-8") as f:
        dev_cases = json.load(f)

    total_recall, total_mrr, total_precision = 0.0, 0.0, 0.0
    failures = []

    print(f"\n🔍 Running Layer 1 Retriever Eval on {len(dev_cases)} Dev Cases (top_k={top_k})...\n")
    print(f"{'Case ID':<18} | {'Recall':<6} | {'MRR':<6} | {'Precision':<9} | {'Gold Doc':<20} | {'Retrieved Docs'}")
    print("-" * 105)

    for case in dev_cases:
        chunks = retrieve(case["question"], top_k=top_k)
        scores = score_single_retrieval(chunks, case["gold_doc_ids"])

        total_recall += scores["recall"]
        total_mrr += scores["mrr"]
        total_precision += scores["precision"]

        print(
            f"{case['case_id']:<18} | "
            f"{scores['recall']:<6.2f} | "
            f"{scores['mrr']:<6.2f} | "
            f"{scores['precision']:<9.2f} | "
            f"{case['doc_name']:<20} | "
            f"{scores['retrieved_docs']}"
        )

        if scores["recall"] == 0.0:
            failures.append((case, scores["retrieved_docs"]))

    n = len(dev_cases)
    print("-" * 105)
    print(f"📊 BASELINE V1 SUMMARY (Average across {n} Dev Cases):")
    print(f"   • Mean Recall@{top_k} (Coverage)  : {total_recall / n:.2%}")
    print(f"   • Mean MRR@{top_k} (Ranking)      : {total_mrr / n:.2%}")
    print(f"   • Mean Precision@{top_k} (Clean)  : {total_precision / n:.2%}")
    print(f"   • Complete Misses (Recall = 0)  : {len(failures)} / {n} questions failed completely!\n")

if __name__ == "__main__":
    run_retriever_eval(top_k=3)