"""
Compare saved eval runs side by side.

Usage (from the project root):
    python evals/compare.py              # all runs
    python evals/compare.py --split dev  # only dev runs
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS = ["recall", "mrr", "precision", "ctx_recall", "ctx_precision"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=None)
    args = p.parse_args()

    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(RESULTS_DIR.glob("*.json"))]
    if args.split:
        runs = [r for r in runs if r["config"]["split"] == args.split]
    if not runs:
        print("No runs found in results/. Run evals/run_eval.py first.")
        return

    header = f"{'run_id':<38} {'n':>3} {'k':>2} " + " ".join(f"{m:>13}" for m in METRICS) + "  note"
    print(header)
    print("-" * len(header))
    for r in runs:
        s, c = r["summary"], r["config"]
        cells = " ".join(f"{s[m]:>13.2%}" if m in s else f"{'-':>13}" for m in METRICS)
        print(f"{r['run_id']:<38} {c['n_cases']:>3} {c['k']:>2} {cells}  {c.get('note', '')}")

    print("\nOnly compare runs with the same split, n and k. Different settings = different test.")


if __name__ == "__main__":
    main()
