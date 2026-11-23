"""
Experiment H5: one search per year (idea from Dagmawi Zewdu, LinkedIn).

Same as retriever_8_filter_fix, ONE change:
    If the question names 2+ years AND those years live in different reports,
    search each year's report(s) separately with its own top_k, then join.
    So 'compare 2015 vs 2016' gets top_k chunks from 2015 AND top_k from 2016,
    instead of one ranked list where one year can take every slot.

Everything else (one year, no year, or both years in one report) is unchanged.

Note: a two-year question now returns up to 2 x top_k chunks. Recall can rise
partly just because more chunks are returned; keep that in mind when comparing.
"""

import chromadb

from src.retriever_6_bge import COLLECTION_NAME, embed_query
from src.retriever_7_filter import DB_PATH, report_year
from src.retriever_8_filter_fix import extract_filters


def _search(collection, query_embedding, top_k: int, reports: list[str] | None) -> list[dict]:
    query = {"query_embeddings": [query_embedding], "n_results": top_k, "include": ["documents", "metadatas"]}
    if reports:
        query["where"] = {"doc_name": {"$in": reports}}
    results = collection.query(**query)
    out = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        out.append({
            "chunk_id": results["ids"][0][i],
            "doc_id": meta["doc_id"],
            "doc_name": meta["doc_name"],
            "page_num": meta.get("page_num", "N/A"),
            "text": results["documents"][0][i],
        })
    return out


def reports_by_year(filters: dict) -> dict[str, list[str]]:
    """{'2015': ['ADOBE_2015_10K'], '2016': ['ADOBE_2016_10K']}: only the years the question asked for."""
    groups: dict[str, list[str]] = {}
    for r in filters["reports"]:
        y = report_year(r)
        if y in filters["years"]:
            groups.setdefault(y, []).append(r)
    return groups


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(name=COLLECTION_NAME)
    filters = extract_filters(question)
    emb = embed_query(question)

    groups = reports_by_year(filters)
    if len(groups) >= 2:  # H5: one search per year
        retrieved = []
        for year in sorted(groups):
            retrieved += _search(collection, emb, top_k, groups[year])
        return retrieved

    return _search(collection, emb, top_k, filters["reports"] or None)  # same as retriever_8
