"""
Experiment H4b: fix Step 1 (reading company + year from the question).

Same as retriever_7_filter (H4) except extract_filters(). Bugs it fixes,
found on the frozen test run (20261122_215625_retriever_7_filter_test):
    1. Case:     'AMCOR' did not match the stored name 'Amcor'.
    2. Year:     'FY22' was not read as 2022 (only 4-digit years were).
    3. Nickname: 'AMEX' did not match 'American Express'.

Name matching rule: ignore case, BUT the first letter in the question must be
uppercase. 'AMCOR', 'Amcor', 'PayPal' match; the word 'block' does not
select the company 'Block'.

Note: the frozen test set must not be used to measure this fix (we already
looked at its failures). Check it with evals/test_3_extract_filters.py and dev.
"""

import re

import chromadb

from src.retriever_6_bge import COLLECTION_NAME, embed_query
from src.retriever_7_filter import DB_PATH, known_reports, name_variants, report_year

# Nicknames that aren't a spelling of the stored name. Keep this short and explicit.
NICKNAMES = {
    "American Express": ["AMEX", "Amex"],
}


def mentions(name: str, question: str) -> bool:
    """True if `name` appears as a whole word, any case, starting with a capital."""
    for m in re.finditer(rf"(?<!\w){re.escape(name)}(?!\w)", question, flags=re.IGNORECASE):
        if not m.group(0)[0].islower():
            return True
    return False


def find_years(question: str) -> list[str]:
    four_digit = re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)                    # 2018, FY2018
    two_digit = ["20" + y for y in re.findall(r"\bFY\s?'?(\d{2})(?!\d)", question)]  # FY22, FY '22
    return sorted(set(four_digit + two_digit))


def extract_filters(question: str) -> dict:
    """Step 1: what company and year(s) does the question ask about?"""
    companies = [
        company for company in known_reports()
        if any(mentions(v, question) for v in name_variants(company) + NICKNAMES.get(company, []))
    ]
    years = find_years(question)

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
        query["where"] = {"doc_name": {"$in": filters["reports"]}}

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
