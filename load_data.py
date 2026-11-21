import json
from pathlib import Path
from datasets import load_dataset

def setup_financebench_data():
    print("⏳ Downloading PatronusAI/financebench from Hugging Face...")
    dataset = load_dataset("PatronusAI/financebench", split="train")
    print(f"✅ Successfully loaded {len(dataset)} rows!\n")

    # 1. Print the column names and Sample #0 so we understand the exact structure
    first_row = dataset[0]
    print("--- DATASET COLUMNS ---")
    print(list(first_row.keys()))

    print("\n--- SAMPLE CASE #0 ---")
    print(f"Company      : {first_row.get('company')}")
    print(f"Document     : {first_row.get('doc_name')}")
    print(f"Question Type: {first_row.get('question_type')}")
    print(f"Question     : {first_row.get('question')}")
    print(f"Gold Answer  : {first_row.get('answer')}")

    # 2. Extract clean documents for our Retriever Corpus and split into Dev (20) & Test (30)
    corpus_docs = []
    dev_cases = []
    test_cases = []

    for i, row in enumerate(dataset):
        case_id = row.get("financebench_id", f"fb_{i:03d}")
        evidences = row.get("evidence", [])

        # Extract the gold evidence pages/paragraphs for this question
        gold_texts = []
        gold_doc_ids = []

        if isinstance(evidences, list):
            for idx, ev in enumerate(evidences):
                if isinstance(ev, dict):
                    # FinanceBench provides 'evidence_text' and 'evidence_text_full_page'
                    text = ev.get("evidence_text_full_page") or ev.get("evidence_text") or ""
                    text = text.strip()
                    if text:
                        doc_id = f"{case_id}_ev_{idx}"
                        page_num = ev.get("evidence_page_num", "unknown")
                        
                        gold_texts.append(text)
                        gold_doc_ids.append(doc_id)

                        corpus_docs.append({
                            "doc_id": doc_id,
                            "company": row.get("company", "Unknown"),
                            "doc_name": row.get("doc_name", "Unknown"),
                            "page_num": page_num,
                            "text": text
                        })

        # Skip if for any reason a row has no evidence text
        if not gold_texts:
            continue

        clean_case = {
            "case_id": case_id,
            "company": row.get("company", "Unknown"),
            "doc_name": row.get("doc_name", "Unknown"),
            "question_type": row.get("question_type", "Unknown"),
            "question": row.get("question", ""),
            "expected_output": row.get("answer", ""),
            "justification": row.get("justification", ""),
            "gold_doc_ids": gold_doc_ids,
            "gold_evidence_texts": gold_texts
        }

        if len(dev_cases) < 20:
            dev_cases.append(clean_case)
        elif len(test_cases) < 30:
            test_cases.append(clean_case)

    # 3. Save everything into the data/ folder
    Path("data").mkdir(exist_ok=True)

    with open("data/corpus_docs.json", "w", encoding="utf-8") as f:
        json.dump(corpus_docs, f, indent=2, ensure_ascii=False)

    with open("data/dev_20.json", "w", encoding="utf-8") as f:
        json.dump(dev_cases, f, indent=2, ensure_ascii=False)

    with open("data/frozen_test_30.json", "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2, ensure_ascii=False)

    print("\n--- FILES SAVED IN data/ ---")
    print(f"📁 data/corpus_docs.json    : {len(corpus_docs)} source financial pages")
    print(f"📁 data/dev_20.json         : {len(dev_cases)} development test cases")
    print(f"📁 data/frozen_test_30.json : {len(test_cases)} frozen test cases")

if __name__ == "__main__":
    setup_financebench_data()