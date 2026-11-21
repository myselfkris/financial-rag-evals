import os
import json
import sys
from pathlib import Path

from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric
from deepeval.models import DeepSeekModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.retriever_3 import retrieve

def run_deepeval_retriever(num_cases: int = 5, top_k: int = 3):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("❌ Please set $env:DEEPSEEK_API_KEY in PowerShell first!")

    # DeepSeek Chat (fast, cheap, strong JSON judge)
    deepseek_judge = DeepSeekModel(
        model="deepseek-flash",
        api_key=api_key,
        temperature=0
    )

    recall_metric = ContextualRecallMetric(
        threshold=0.8,
        model=deepseek_judge
    )
    precision_metric = ContextualPrecisionMetric(
        threshold=0.8,
        model=deepseek_judge
    )

    with open("data/dev_20.json", "r", encoding="utf-8") as f:
        dev_cases = json.load(f)[:num_cases]

    print(f"\n🧠 Running DeepEval (DeepSeek Judge) on first {len(dev_cases)} V1 Cases...\n")

    total_recall, total_precision = 0.0, 0.0

    for i, case in enumerate(dev_cases, 1):
        chunks = retrieve(case["question"], top_k=top_k)
        chunk_texts = [c["text"] for c in chunks]
        retrieved_docs = [c["doc_name"] for c in chunks]

        gold_facts = f"For {case['company']} ({case['doc_name']}): {case['expected_output']}"
        if case.get("justification"):
            gold_facts += f" ({case['justification']})"

        test_case = LLMTestCase(
            input=case["question"],
            actual_output="",
            retrieval_context=chunk_texts,
            expected_output=gold_facts
        )

        recall_metric.measure(test_case)
        precision_metric.measure(test_case)

        total_recall += recall_metric.score
        total_precision += precision_metric.score

        print(f"[{i}] {case['case_id']} | Needed: {case['doc_name']} | Pulled: {retrieved_docs}")
        print(f"    • Fact Recall   : {recall_metric.score:.2f} -> {recall_metric.reason}")
        print(f"    • Fact Precision: {precision_metric.score:.2f} -> {precision_metric.reason}\n")

    n = len(dev_cases)
    print("-" * 90)
    print(f"📊 DEEPEVAL V1 SUMMARY ({n} Cases):")
    print(f"   • Mean Contextual Recall (Facts Found) : {total_recall / n:.2%}")
    print(f"   • Mean Contextual Precision (Ranking)  : {total_precision / n:.2%}")
    print("-" * 90)

if __name__ == "__main__":
    run_deepeval_retriever(num_cases=5, top_k=3)