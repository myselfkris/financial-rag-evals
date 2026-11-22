"""
Experiment H4: filter by company + year taken from the QUESTION, then search.
(Idea from a LinkedIn comment.)

ONE change vs retriever_6_bge (H2, current best):
    Step 1: read the company name and year(s) from the question text.
    Step 2: only search chunks whose LABEL (metadata) matches those reports.
    Step 3: rank as before (same BGE model, same chunks, same top_k).

No re-indexing needed: this reuses H2's collection. The company and report
name are already stored as labels on every chunk.

Not leakage: everything comes from the user's question plus the list of
reports we indexed. Nothing comes from the test file's answer key.

Fallbacks (so a bad extraction doesn't block search entirely):
    no company found          -> search everything (same as H2)
    company found, no year    -> filter by company only
    company + year, no report -> filter by company only
"""

import json
import re
from functools import lru_cache

import chromadb

from src.retriever_6_bge import COLLECTION_NAME, embed_query

DB_PATH = "data/chroma_db"


@lru_cache(maxsize=1)
def known_reports() -> dict[str, list[str]]:
    """company -> list of report names we indexed, e.g. '3M' -> ['3M_2018_10K', ...]."""
    with open("data/corpus_docs.json", "r", encoding="utf-8") as f:
        docs = json.load(f)
    reports: dict[str, set[str]] = {}
    for d in docs:
        reports.setdefault(d["company"], set()).add(d["doc_name"])
    return {company: sorted(names) for company, names in reports.items()}


def report_year(doc_name: str) -> str | None:
    """'3M_2023Q2_10Q' -> '2023'."""
    m = re.search(r"_(\d{4})", doc_name)
    return m.group(1) if m else None


def name_variants(company: str) -> list[str]:
    """'AES Corporation' -> ['AES Corporation', 'AES']."""
    variants = [company]
    short = re.sub(r"\s+(Corporation|Corp\.?|Inc\.?|Company|Co\.?)$", "", company)
    if short != company:
        variants.append(short)
    return variants


def extract_filters(question: str) -> dict:
    """Step 1: what company and year(s) does the question ask about?"""
    # Case-sensitive on purpose: 'Block' (the company) vs 'block' (a word).
    companies = [
        company for company in known_reports()
        if any(re.search(rf"(?<!\w){re.escape(v)}(?!\w)", question) for v in name_variants(company))
    ]
    years = sorted(set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)))  # catches '2018' and 'FY2018'

    reports = []
    for company in companies:
        company_reports = known_reports()[company]
        if years:
            matched = [r for r in company_reports if report_year(r) in years]
            reports += matched or company_reports  # year not found -> keep all of this company
        else:
            reports += company_reports

    return {"companies": companies, "years": years, "reports": reports}


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    filters = extract_filters(question)
    query = {
        "query_embeddings": [embed_query(question)],
        "n_results": top_k,
        "include": ["documents", "metadatas"],
    }
    if filters["reports"]:
        query["where"] = {"doc_name": {"$in": filters["reports"]}}  # Step 2: only open these 'boxes'

    results = collection.query(**query)

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        retrieved.append({
            "chunk_id": results["ids"][0][i],
            "doc_id": meta["doc_id"],
            "doc_name": meta["doc_name"],
            "page_num": meta.get("page_num", "N/A"),
            "text": results["documents"][0][i],
        })
    return retrieved
