"""
Check Step 1 of retriever_7_filter on its own:
did we read the right company and year from each question?

If this step is wrong, the filter hides the right report and nothing after it
can recover. So it gets its own eval. Free: no model, no API calls.

Usage (from the project root):
    python evals/eval_extraction.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.chdir(ROOT)

from src.retriever_7_filter import extract_filters


def main():
    with open("data/dev_20.json", encoding="utf-8") as f:
        cases = json.load(f)

    company_ok = report_kept = narrowed = 0
    print(f"{'case_id':<22} {'company?':<9} {'report kept?':<13} {'#reports':<9} question")
    print("-" * 110)
    for c in cases:
        f = extract_filters(c["question"])
        got_company = c["company"] in f["companies"]
        kept = c["doc_name"] in f["reports"] if f["reports"] else True  # no filter = nothing hidden
        company_ok += got_company
        report_kept += kept
        narrowed += bool(f["reports"])
        print(f"{c['case_id']:<22} {'yes' if got_company else 'NO':<9} {'yes' if kept else 'NO':<13} "
              f"{len(f['reports']) or 'all':<9} {c['question'][:55]}")

    n = len(cases)
    print("-" * 110)
    print(f"Company read correctly        : {company_ok}/{n}")
    print(f"Right report kept after filter: {report_kept}/{n}   <- if NO, the filter hid the answer")
    print(f"Questions where a filter was used: {narrowed}/{n}")


if __name__ == "__main__":
    main()
