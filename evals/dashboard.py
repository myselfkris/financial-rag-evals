"""
Build a local, single-file dashboard of every saved eval run.

Usage (from the project root):
    python evals/dashboard.py          # writes results/dashboard.html, then open it in a browser

Re-run after every new eval. Three views on one page:
    1. Progress  - a metric across experiments, one line per split (dev / dev2 / test)
    2. Compare   - pick two runs on the same split: which cases got fixed, broke, stayed missing
    3. Case      - click a case: question, Step 1 output, gold page vs the 3 retrieved pages
"""

import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # retrievers read relative paths like data/corpus_docs.json

RESULTS_DIR = ROOT / "results"
OUT = RESULTS_DIR / "dashboard.html"
SPLIT_FILES = {"dev": "data/dev_20.json", "dev2": "data/dev2_20.json", "test": "data/frozen_test_30.json"}

# Short experiment names, in the order they were run.
EXPERIMENTS = {
    "retriever_3": "Baseline",
    "retriever_4_header": "H1",
    "retriever_5_page_header": "H1b",
    "retriever_6_bge": "H2",
    "retriever_7_filter": "H4",
    "retriever_8_filter_fix": "H4b",
    "retriever_9_per_year": "H5",
}
SNIPPET_CHARS = 1500


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def labels_for(run_id, fl):
    """Hand labels for one run, wherever failure_labels.json keeps them."""
    if run_id in fl.get("runs", {}):
        return fl["runs"][run_id].get("labels", {})
    for entry in fl.values():
        if isinstance(entry, dict) and entry.get("run_id") == run_id:
            return entry.get("labels", {})
    return {}


def step1_function(retriever):
    """extract_filters() if this retriever has a Step 1, else None."""
    try:
        mod = importlib.import_module(f"src.{retriever}")
        return getattr(mod, "extract_filters", None)
    except Exception as e:  # missing module or dependency: show the run without Step 1
        print(f"  (no Step 1 for {retriever}: {e})")
        return None


def main():
    fl = load_json("evals/failure_labels.json")
    corpus = {d["doc_id"]: d for d in load_json("data/corpus_docs.json")}

    cases_info, pages = {}, {}
    for split, path in SPLIT_FILES.items():
        if not Path(path).exists():
            continue
        for c in load_json(path):
            cases_info[c["case_id"]] = {
                "split": split,
                "question": c["question"],
                "expected": c["expected_output"],
                "type": c["question_type"],
                "gold_doc_name": c["doc_name"],
                "gold_doc_ids": c["gold_doc_ids"],
            }
            for gid in c["gold_doc_ids"]:
                pages[gid] = None  # filled from corpus below

    runs = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        r = load_json(f)
        cfg = r["config"]
        run_id = r.get("run_id", f.stem)
        extract = step1_function(cfg["retriever"])
        labels = labels_for(run_id, fl)
        cases = []
        for c in r["cases"]:
            lab = labels.get(c["case_id"], {})
            retrieved = [{"doc_id": x["doc_id"], "doc_name": x["doc_name"], "chunk_id": x.get("chunk_id", "")}
                         for x in c["retrieved"]]
            for x in retrieved:
                pages[x["doc_id"]] = None
            step1 = None
            if extract:
                s = extract(c["question"])
                step1 = {"companies": s["companies"], "years": s["years"], "reports": s["reports"]}
            cases.append({
                "id": c["case_id"],
                "recall": c["recall"], "mrr": c["mrr"], "precision": c["precision"],
                "retrieved": retrieved,
                "label": lab.get("label", ""),
                "notes": lab.get("notes", ""),
                "step1": step1,
                # LLM-judge layer (only on runs made with --judge)
                "ctx_recall": c.get("ctx_recall"),
                "ctx_precision": c.get("ctx_precision"),
                "ctx_recall_reason": c.get("ctx_recall_reason", ""),
                "ctx_precision_reason": c.get("ctx_precision_reason", ""),
            })
        runs.append({
            "id": run_id,
            "retriever": cfg["retriever"],
            "exp": EXPERIMENTS.get(cfg["retriever"], cfg["retriever"]),
            "split": cfg["split"],
            "note": cfg.get("note", ""),
            "contaminated": "CONTAMINATED" in cfg.get("note", "").upper(),
            "summary": r["summary"],
            "n": len(cases),
            "cases": cases,
        })

    for doc_id in pages:
        d = corpus.get(doc_id)
        if d:
            text = " ".join(d["text"].split())
            pages[doc_id] = {"doc_name": d["doc_name"], "page": d.get("page_num", "?"), "text": text[:SNIPPET_CHARS]}

    data = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiments": list(dict.fromkeys(EXPERIMENTS.values())),
        "runs": runs,
        "cases": cases_info,
        "pages": pages,
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    OUT.write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(runs)} runs). Open it in your browser.")


if __name__ == "__main__":
    main()
