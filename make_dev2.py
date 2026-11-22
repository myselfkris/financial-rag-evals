"""
Build a second, fairer dev set: data/dev2_20.json

Why: load_data.py took the first 20 rows for dev and the next 30 for test.
FinanceBench is sorted by company, so dev got 3M/Adobe/AES/Activision/Amazon
and test got Amcor/AMD/AmEx/... Dev could not warn us about test's problems.

How: from the 100 questions in NEITHER dev_20 nor frozen_test_30, pick 20
at random, spread across companies (round-robin: one per company, then a
second per company, ...). Fixed seed, so the same set comes out every time.

Their evidence pages are already in corpus_docs.json (load_data.py indexed
the evidence of all 150 questions), so no re-indexing is needed.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
N = 20


def clean_case(row: dict) -> dict | None:
    """Same format and doc_id scheme as load_data.py."""
    case_id = row["financebench_id"]
    gold_texts, gold_doc_ids = [], []
    for idx, ev in enumerate(row.get("evidence") or []):
        text = (ev.get("evidence_text_full_page") or ev.get("evidence_text") or "").strip()
        if text:
            gold_texts.append(text)
            gold_doc_ids.append(f"{case_id}_ev_{idx}")
    if not gold_texts:
        return None
    return {
        "case_id": case_id,
        "company": row.get("company", "Unknown"),
        "doc_name": row.get("doc_name", "Unknown"),
        "question_type": row.get("question_type", "Unknown"),
        "question": row.get("question", ""),
        "expected_output": row.get("answer", ""),
        "justification": row.get("justification", ""),
        "gold_doc_ids": gold_doc_ids,
        "gold_evidence_texts": gold_texts,
    }


def main():
    used = {c["case_id"] for f in ["data/dev_20.json", "data/frozen_test_30.json"]
            for c in json.load(open(f, encoding="utf-8"))}
    ds = load_dataset("PatronusAI/financebench", split="train")

    by_company = defaultdict(list)
    for row in ds:
        if row["financebench_id"] in used:
            continue
        case = clean_case(row)
        if case:
            by_company[case["company"]].append(case)

    rng = random.Random(SEED)
    companies = sorted(by_company)
    rng.shuffle(companies)
    for c in companies:
        rng.shuffle(by_company[c])

    picked = []
    while len(picked) < N and any(by_company.values()):
        for c in companies:
            if by_company[c] and len(picked) < N:
                picked.append(by_company[c].pop())

    Path("data/dev2_20.json").write_text(json.dumps(picked, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(picked)} cases from {len({c['company'] for c in picked})} companies to data/dev2_20.json")


if __name__ == "__main__":
    main()
