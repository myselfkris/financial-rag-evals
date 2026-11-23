# financial-rag-evals

A RAG system for questions about company financial filings (10-Ks, 10-Qs, earnings reports), built **evaluation-first**: every change is a single experiment, scored on fixed question sets, and kept or dropped based on the numbers.

Most RAG demos show that a system *answers*. This project measures *how often it answers correctly, where it fails, and why*, one layer at a time.

## Evaluation roadmap

A RAG answer can go wrong in the retriever (wrong pages fetched) or in the generator (right pages, wrong answer). Testing the whole thing at once hides which one broke, so each layer gets its own eval.

| # | Layer | Question it answers | Status |
|---|---|---|---|
| 1 | **Retriever** | Did we fetch the page that contains the answer? | ✅ Done: 6 experiments, frozen test run |
| 2 | **Generator** | Given the *correct* pages, does the LLM answer correctly and stick to them? | 🟡 In progress: generator and runner built, hand-labelling next |
| 3 | **Pipeline** | With *retrieved* pages, is the final answer right? When wrong, was it retrieval or generation? | ⏳ Planned |
| 4 | **Application** | Is it usable in practice: latency, cost per question, refusing when the answer isn't there? | ⏳ Planned |

## Dataset

[FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) (Patronus AI): 150 questions about public-company filings, each with the gold answer and the page(s) that prove it.

- **Corpus:** the evidence pages of all 150 questions: 189 pages (168 unique) from 84 filings across 32 companies.
- **Question sets:**

| Set | File | Size | How it was built |
|---|---|---|---|
| `dev` | `data/dev_20.json` | 20 | First 20 questions. Used for tuning. |
| `dev2` | `data/dev2_20.json` | 20 | Random, spread across companies (seed 42). Built because `dev` turned out to be biased (see below). |
| `test` | `data/frozen_test_30.json` | 30 | Frozen. Only for final release checks. |

The eval runner refuses to run on `test` unless you pass `--i-am-releasing`, so the test set can't be used for tuning by accident.

## Retriever eval (Layer 1)

### Metrics

Scored at **page level**, with top-k = 3 chunks:

- **Recall@3:** share of gold pages found in the top 3.
- **MRR:** how high the first correct page ranks (1st = 1.0, 2nd = 0.5, 3rd = 0.33).
- **Precision@3:** share of the 3 chunks that come from a gold page.
- **Complete misses:** questions where no gold page was retrieved at all.
- **Latency p50 / p95:** median and slow-case retrieval time. Averages are distorted by one-off model loading, so they are not reported.
- **Optional LLM judge:** DeepEval contextual recall and precision (judge: DeepSeek), to check whether the retrieved *text* actually supports the answer. Cases where code and judge disagree are flagged for review.

Pages are matched on **text**, not ID: FinanceBench gives each question its own ID per page, so one page cited by two questions has two IDs. Matching on ID would mark correct retrievals as wrong.

Every result file stores a hash of the retriever code and of the question set, so each number traces back to exactly what produced it.

### Experiments

Each experiment changes **one thing** from the previous best.

| ID | Change | Retriever file |
|---|---|---|
| Baseline | 1000-char chunks, 200 overlap, MiniLM embeddings | `src/retriever_3.py` |
| H1 | Add a company and filing header to every chunk | `src/retriever_4_header.py` |
| H1b | Also add the page's top lines (table name, units, years) | `src/retriever_5_page_header.py` |
| H2 | Swap embeddings: MiniLM → `BAAI/bge-base-en-v1.5` | `src/retriever_6_bge.py` |
| H4 | Read company and year from the question, search only matching filings | `src/retriever_7_filter.py` |
| H4b | Fix company/year reading: case (`AMCOR`), short years (`FY22`), nicknames (`AMEX`) | `src/retriever_8_filter_fix.py` |
| H5 | For questions about two years, search each year separately | `src/retriever_9_per_year.py` |

(`src/retriever.py` and `src/retriever_2_chunk_change.py` are early prototypes, before the eval runner existed.)

### Results on `dev` (20 questions)

| Experiment | Recall@3 | MRR | Precision@3 | Complete misses |
|---|---|---|---|---|
| Baseline | 0.60 | 0.48 | 0.35 | 6 |
| H1 company header | 0.60 | 0.62 | 0.48 | 6 |
| H1b + page header | 0.63 | 0.62 | 0.48 | 6 |
| H2 BGE embeddings | 0.68 | 0.78 | 0.62 | 3 |
| H4 company/year filter | **0.77** | **0.88** | **0.77** | **1** |

### Results on `dev2` (20 questions, fairer sample)

| Experiment | Recall@3 | MRR | Precision@3 | Complete misses |
|---|---|---|---|---|
| H4 | 0.80 | 0.82 | 0.82 | 3 |
| H4b | 0.80 | 0.82 | 0.82 | 3 |
| H5 per-year search | 0.80 | 0.75 | 0.77 | 3 |

### Frozen test set (30 questions)

| Run | Recall@3 | MRR | Precision@3 | Misses | p50 / p95 latency |
|---|---|---|---|---|---|
| **H4: release score** | **0.73** | **0.77** | **0.62** | 7 | 163 / 219 ms |
| H4b | 0.77 | 0.80 | 0.64 | 6 | 97 / 112 ms |

**The H4b test score is marked contaminated and is not the release score.** H4b's fixes were chosen *after* looking at H4's failures on the test set, so the test set is no longer an unbiased check for it. H4b was verified with unit tests (`evals/test_3_extract_filters.py`) and on `dev2` instead.

The LLM judge was run on the H4b test set as a sanity check on the code metrics: contextual recall was 0.74 vs 0.77 from the code, and the two disagreed on 2 of 30 questions.

### What the experiments showed

- **H1 (company header):** wrong-company retrievals among the misses went from 4 to 0, but recall didn't move, because new wrong-year and wrong-page failures appeared. MRR went up: right pages started ranking higher.
- **H2 (better embeddings):** the biggest single jump in ranking (MRR 0.62 → 0.78) and halved complete misses. The small model could tell topics apart but not companies or years.
- **H4 (filter by company and year):** the biggest overall gain on `dev`. But on the frozen test set it scored lower (recall 0.73 vs 0.77), which exposed a problem with `dev` itself:
- **`dev` was biased.** FinanceBench is sorted by company, so `dev` (first 20) covered only 5 companies starting with A or 3, and `test` (next 30) covered different ones. `dev` couldn't warn about test's failures (e.g. `AMCOR` in capitals, `FY22`). `dev2` samples randomly across companies to fix this.
- **H5 (search each year separately):** recall unchanged, MRR dropped (0.82 → 0.75). **Not adopted.**

Every miss on each run was hand-labelled with what went wrong (`company_mismatch`, `year_mismatch`, `right_doc_wrong_page`, `right_company_wrong_report`) and a suspected cause. See `evals/failure_labels.json`.

## Generator eval (Layer 2, in progress)

`src/generator.py` is a deliberately plain baseline prompt (`v1_generic`): answer only from the context, otherwise say "I don't know." Rules about formulas, rounding or reading tables get added only if error analysis shows they're needed.

`evals/run_generator.py` runs it in two modes:

- `--context gold`: feed the **correct** evidence pages. Tests the generator alone.
- `--context retrieved`: feed what the retriever returned. Tests the whole pipeline (Layer 3).

Answers are saved with empty `my_correct` / `my_faithful` fields to be **labelled by hand first**, before building an LLM judge, so the judge can later be checked against human labels.

## Repo structure

```
load_data.py              Download FinanceBench, build corpus + dev/test sets
make_dev2.py              Build the fairer dev2 set
src/
  retriever_*.py          One file per experiment (see table above)
  generator.py            Baseline answer generator
evals/
  run_eval.py             Retriever eval runner (code metrics + optional LLM judge)
  run_generator.py        Generator / pipeline runner
  compare.py              Compare saved runs side by side
  dashboard.py            Build a local HTML dashboard of all runs
  eval_extraction.py      Eval for the company/year reading step alone
  test_3_extract_filters.py  Unit tests for company/year reading
  failure_labels.json     Hand-labelled failures per run
data/                     Question sets and corpus (vector index is rebuilt locally)
results/                  One JSON file per eval run
```

## How to run

```bash
python -m venv venv
venv\Scripts\activate          # Windows; on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Build the vector index for the current best retriever (downloads BGE, ~440 MB)
python -m src.retriever_6_bge

# Score a retriever on dev
python evals/run_eval.py --retriever retriever_8_filter_fix --split dev2 --note "what I changed"

# Compare runs / open the dashboard
python evals/compare.py --split dev2
python evals/dashboard.py      # writes results/dashboard.html

# Unit tests
python -m pytest evals/test_3_extract_filters.py -v
```

The LLM judge (`--judge`) and the generator need a DeepSeek API key in a `.env` file:

```
DEEPSEEK_API_KEY=your_key_here
```

## Limitations

- **Small question sets.** One question is 5% on `dev` and 3.3% on `test`, so differences of one or two questions are noise, not progress.
- **Easy corpus.** The corpus contains only the 189 evidence pages, not the full filings (hundreds of pages each). Retrieval over full filings would be much harder, and the scores here are an upper bound for that setting.
- **Hand-built name matching.** The company/year filter uses rules plus a short nickname list, which will miss names it hasn't seen.

## Data and credits

Questions and evidence come from [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) by Patronus AI; see the dataset page for its license. The company/year filter (H4) and per-year search (H5) ideas came from comments on LinkedIn.
