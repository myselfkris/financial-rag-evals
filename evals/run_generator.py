"""
Generator runner: produces answers and saves them. It does NOT score them yet.

Two context modes (Topic 5):
    --context gold       Run B: feed the gold evidence text (perfect retrieval)
                         -> tests the GENERATOR alone
    --context retrieved  Run A: feed what the retriever returns
                         -> tests the whole PIPELINE

Usage (from the project root):
    python evals/run_generator.py --context gold --n 3
    python evals/run_generator.py --context retrieved --retriever retriever_8_filter_fix
"""

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.run_eval import SPLITS, file_hash, load_page_texts, score_ir  # noqa: E402
from src import generator  # noqa: E402

RESULTS_DIR = ROOT / "results"


def gold_chunks(case: dict) -> list[dict]:
    return [{"text": t, "doc_name": case["doc_name"]} for t in case["gold_evidence_texts"]]


def main():
    p = argparse.ArgumentParser(description="Generate answers and save them (no scoring).")
    p.add_argument("--context", choices=["gold", "retrieved"], required=True)
    p.add_argument("--retriever", default="retriever_8_filter_fix")
    p.add_argument("--split", choices=["dev", "dev2"], default="dev")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--n", type=int, default=None, help="only the first N cases")
    p.add_argument("--note", default="")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    with open(SPLITS[args.split], encoding="utf-8") as f:
        cases = json.load(f)[: args.n]

    retriever = None
    if args.context == "retrieved":
        retriever = importlib.import_module(f"src.{args.retriever}")
        pages = load_page_texts()

    print(f"\nGenerator {generator.PROMPT_VERSION} | model: {args.model} | "
          f"context: {args.context} | split: {args.split} | cases: {len(cases)}\n")

    rows = []
    for i, case in enumerate(cases, 1):
        if retriever:
            chunks = retriever.retrieve(case["question"], top_k=args.k)
        else:
            chunks = gold_chunks(case)

        out = generator.generate(case["question"], chunks, model=args.model)

        row = {
            "case_id": case["case_id"],
            "question_type": case.get("question_type"),
            "question": case["question"],
            "gold_answer": case["expected_output"],
            "answer": out["answer"],
            "context_chunks": [{"doc_name": c.get("doc_name"), "page_num": c.get("page_num"),
                                "text": c["text"]} for c in chunks],
            "latency_ms": out["latency_ms"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            # Fill these in BY HAND before any judge exists (Topic 4: label blind).
            "my_correct": "",    # "yes" / "no"
            "my_faithful": "",   # "yes" / "no"
            "my_note": "",
        }
        if retriever:
            row["recall"] = score_ir(chunks, case["gold_doc_ids"], pages)["recall"]
        rows.append(row)

        preview = out["answer"].replace("\n", " ")[:90]
        print(f"[{i:>2}] {case['case_id']:<22} {preview}")

    run_id = (datetime.now().strftime("%Y%m%d_%H%M%S")
              + f"_generator_{generator.PROMPT_VERSION}_{args.context}_{args.split}")
    run = {
        "run_id": run_id,
        "config": {
            "prompt_version": generator.PROMPT_VERSION,
            "generator_file_hash": file_hash("src/generator.py"),
            "model": args.model,
            "context": args.context,
            "retriever": args.retriever if retriever else None,
            "k": args.k if retriever else None,
            "split": args.split,
            "split_file_hash": file_hash(SPLITS[args.split]),
            "n_cases": len(rows),
            "note": args.note,
        },
        "cases": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
