# financial-rag-evals

A RAG system that answers questions about company financial filings, built **evaluation-first**: every change is one experiment, scored on fixed question sets, and kept only if the numbers improve.

## Results

Retriever, top-3 chunks, on [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) questions:

| | Recall@3 | MRR |
|---|---|---|
| Baseline (dev) | 0.60 | 0.48 |
| Best after 4 experiments (dev) | **0.77** | **0.88** |
| **Frozen test set (30 questions)** | **0.73** | **0.77** |

p95 retrieval latency: 219 ms on CPU.

The biggest gains came from better embeddings (BGE) and filtering by the company and year named in the question.

## What makes this different

- **One change per experiment.** Every run saves its code hash and data hash, so every number traces back to exactly what produced it.
- **Every failure is labelled by hand** (wrong company, wrong year, wrong page) before the next fix is chosen.
- **The test set is protected.** The runner blocks it unless you pass `--i-am-releasing`. When a fix was later chosen after seeing test failures, that score was marked *contaminated* and not reported as the result.

Full tables and findings: **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**

## Roadmap

| Layer | What it checks | Status |
|---|---|---|
| 1. Retriever | Did we fetch the page with the answer? | ✅ Done |
| 2. Generator | Given the right pages, is the answer correct? | 🟡 In progress |
| 3. Pipeline | End to end: if wrong, was it retrieval or generation? | ⏳ Planned |
| 4. Application | Latency, cost per question, refusing when unsure | ⏳ Planned |

## Quick start

```bash
pip install -r requirements.txt
python -m src.retriever_6_bge                                    # build the index
python evals/run_eval.py --retriever retriever_8_filter_fix --split dev2
python evals/dashboard.py                                        # results/dashboard.html
```

The LLM judge and generator need `DEEPSEEK_API_KEY` in a `.env` file.

## Limitations

- Small question sets (20–30), so one question moves a score by 3–5%.
- The corpus is the 189 evidence pages, not full filings, so real-world retrieval would be harder.

Data: [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) by Patronus AI.
