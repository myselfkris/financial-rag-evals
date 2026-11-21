"""
Retriever eval runner.

Scores any retriever in src/ on a dataset split and saves the full results to
results/, so different versions can be compared over time.

Usage (from the project root):
    python evals/run_eval.py --retriever retriever_3
    python evals/run_eval.py --retriever retriever_3 --judge          # + DeepEval (costs API calls)
    python evals/run_eval.py --retriever retriever_3 --judge --n 5    # judge only the first 5 cases
    python evals/run_eval.py --retriever retriever_3 --split test --i-am-releasing

A retriever module only needs:
    COLLECTION_NAME : str
    retrieve(question: str, top_k: int) -> list[dict]   # each dict has "doc_id" and "text"
"""

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # retrievers use relative paths like "data/chroma_db"

SPLITS = {
    "dev": "data/dev_20.json",
    "test": "data/frozen_test_30.json",
}
RESULTS_DIR = ROOT / "results"


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def load_page_texts() -> dict[str, str]:
    """doc_id -> page text.

    We match on page TEXT, not doc_id: load_data.py gives each question its own
    doc_id, so one page cited by two questions has two ids. Matching on doc_id
    would mark a correct retrieval as wrong.
    """
    with open("data/corpus_docs.json", encoding="utf-8") as f:
        return {d["doc_id"]: d["text"] for d in json.load(f)}


# --------------------------------------------------------------------------
# Layer 1: page-level IR metrics (free, deterministic)
# Question answered: "did we fetch a chunk from the right page?"
# --------------------------------------------------------------------------
def score_ir(retrieved: list[dict], gold_doc_ids: list[str], pages: dict[str, str]) -> dict:
    gold_pages = {pages[g] for g in gold_doc_ids}
    retrieved_pages = [pages.get(c["doc_id"]) for c in retrieved]

    found = gold_pages.intersection(retrieved_pages)
    recall = len(found) / len(gold_pages) if gold_pages else 0.0

    mrr = 0.0
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in gold_pages:
            mrr = 1.0 / rank
            break

    useful = sum(1 for p in retrieved_pages if p in gold_pages)
    precision = useful / len(retrieved) if retrieved else 0.0

    return {"recall": recall, "mrr": mrr, "precision": precision}


# --------------------------------------------------------------------------
# Layer 2: DeepEval LLM judge (costs API calls)
# Question answered: "does the retrieved TEXT support the expected answer?"
# --------------------------------------------------------------------------
def build_judge_metrics(judge_model: str):
    from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
    from deepeval.models import DeepSeekModel

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("DEEPSEEK_API_KEY not found. Put it in .env or set it in your shell.")

    judge = DeepSeekModel(model=judge_model, api_key=api_key, temperature=0)
    return {
        "ctx_recall": ContextualRecallMetric(threshold=0.8, model=judge, async_mode=False),
        "ctx_precision": ContextualPrecisionMetric(threshold=0.8, model=judge, async_mode=False),
    }


def expected_output_for(case: dict) -> str:
    """What the judge compares the chunks against.

    Same as test_2_retriever_deepeval.py so results stay comparable.
    Known limitation: this is the FINAL answer, which is often computed
    (e.g. "$8.70" vs "8,738" on the page), so ctx_recall can under-score.
    """
    text = f"For {case['company']} ({case['doc_name']}): {case['expected_output']}"
    if case.get("justification"):
        text += f" ({case['justification']})"
    return text


def score_judge(metrics: dict, case: dict, retrieved: list[dict]) -> dict:
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=case["question"],
        actual_output="",
        retrieval_context=[c["text"] for c in retrieved],
        expected_output=expected_output_for(case),
    )
    out = {}
    for name, metric in metrics.items():
        metric.measure(test_case)
        out[name] = metric.score
        out[f"{name}_reason"] = metric.reason
    return out


# --------------------------------------------------------------------------
# Run bookkeeping: every result must be traceable to the exact code + data
# --------------------------------------------------------------------------
def file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def check_collection_exists(name: str):
    import chromadb

    existing = [c.name for c in chromadb.PersistentClient(path="data/chroma_db").list_collections()]
    if name not in existing:
        sys.exit(
            f"Collection '{name}' is not in data/chroma_db (found: {existing}).\n"
            f"Build it first, e.g.: python -m src.<retriever_module>"
        )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Score a retriever and save the results.")
    p.add_argument("--retriever", required=True, help="module name in src/, e.g. retriever_3")
    p.add_argument("--split", choices=SPLITS, default="dev")
    p.add_argument("--k", type=int, default=3, help="top_k chunks to retrieve")
    p.add_argument("--judge", action="store_true", help="also run DeepEval metrics (costs API calls)")
    p.add_argument("--judge-model", default="deepseek-flash")
    p.add_argument("--n", type=int, default=None, help="only the first N cases")
    p.add_argument("--note", default="", help="what you changed / why you're running this")
    p.add_argument("--i-am-releasing", action="store_true",
                   help="required for --split test (the frozen set is for final checks only)")
    args = p.parse_args()

    # Protect the frozen test set: tuning on it makes it useless as a final check.
    if args.split == "test" and not args.i_am_releasing:
        sys.exit("The frozen test set is for final release checks only. "
                 "Tune on --split dev. If this is a release check, add --i-am-releasing.")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    module = importlib.import_module(f"src.{args.retriever}")
    collection = getattr(module, "COLLECTION_NAME", "unknown")
    check_collection_exists(collection)

    with open(SPLITS[args.split], encoding="utf-8") as f:
        cases = json.load(f)[: args.n]
    pages = load_page_texts()
    judge_metrics = build_judge_metrics(args.judge_model) if args.judge else None

    print(f"\nRetriever: {args.retriever} (collection: {collection}) | "
          f"split: {args.split} | cases: {len(cases)} | k={args.k} | judge: {args.judge}\n")

    rows = []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        retrieved = module.retrieve(case["question"], top_k=args.k)
        latency_ms = (time.perf_counter() - t0) * 1000

        row = {
            "case_id": case["case_id"],
            "question_type": case.get("question_type"),
            "question": case["question"],
            "gold_doc_name": case["doc_name"],
            "retrieved": [{"chunk_id": c.get("chunk_id"), "doc_name": c.get("doc_name"),
                           "doc_id": c["doc_id"]} for c in retrieved],
            "latency_ms": round(latency_ms, 1),
            **score_ir(retrieved, case["gold_doc_ids"], pages),
            "failure_label": "",  # fill in by hand during error analysis
        }
        if judge_metrics:
            row.update(score_judge(judge_metrics, case, retrieved))
        rows.append(row)

        line = f"[{i:>2}] {case['case_id']:<22} recall={row['recall']:.2f} mrr={row['mrr']:.2f} prec={row['precision']:.2f}"
        if judge_metrics:
            line += f" | ctx_recall={row['ctx_recall']:.2f} ctx_prec={row['ctx_precision']:.2f}"
        print(line)

    summary = {
        "recall": mean([r["recall"] for r in rows]),
        "mrr": mean([r["mrr"] for r in rows]),
        "precision": mean([r["precision"] for r in rows]),
        "complete_misses": sum(1 for r in rows if r["recall"] == 0),
        "latency_ms_avg": mean([r["latency_ms"] for r in rows]),
    }
    if judge_metrics:
        summary["ctx_recall"] = mean([r["ctx_recall"] for r in rows])
        summary["ctx_precision"] = mean([r["ctx_precision"] for r in rows])
        # Where the two layers disagree is where to look first.
        summary["disagreements"] = [r["case_id"] for r in rows
                                    if (r["recall"] > 0) != (r["ctx_recall"] > 0)]

    run = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.retriever}_{args.split}",
        "config": {
            "retriever": args.retriever,
            "collection": collection,
            "retriever_file_hash": file_hash(f"src/{args.retriever}.py"),
            "split": args.split,
            "split_file_hash": file_hash(SPLITS[args.split]),
            "k": args.k,
            "n_cases": len(rows),
            "judge_model": args.judge_model if args.judge else None,
            "note": args.note,
        },
        "summary": summary,
        "cases": rows,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run['run_id']}.json"
    out_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "-" * 70)
    for key, value in summary.items():
        if key == "latency_ms_avg":
            print(f"  {key:<16}: {value:.0f} ms")
        elif isinstance(value, float):
            print(f"  {key:<16}: {value:.2%}")
        else:
            print(f"  {key:<16}: {value}")
    print(f"\nSaved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
