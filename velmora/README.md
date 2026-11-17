# Velmora Help Center assistant

A small question-answering assistant for the (fictional) Velmora help center. You ask a question in plain English, it finds the right help articles, and replies with sentences taken straight from them, with the sources listed underneath. If the help center doesn't cover the question, it says so instead of guessing.

---

## What's in the folder

```
velmora-qaV2/
├── velmora/              the package
│   ├── data.py           loads the .md articles
│   ├── text.py           cleaning, tokenizing, "a week" -> "7 days"
│   ├── ingest.py         builds the BM25 index and the sentence embeddings
│   ├── retrieval.py      finds the top articles for a question
│   ├── rerank.py         optional rerankers (cross-encoder, semantic, tiny LLM)
│   ├── answer.py         coverage check, conflicts, the answer itself
│   ├── evaluate.py       scores the assistant on eval.jsonl
│   ├── sweep.py          tau x rho grid search with plots
│   └── debug_q.py        look inside one question
├── ai/
│   ├── kb/               the help center articles (KB-01.md ...)
│   └── eval.jsonl        the 40 test questions
├── tests/
│   └── test_evaluate.py  automated tests
├── app.py, index.html    small web interface
├── ask.py                command line
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            
pip install sentence-transformers pytest
pip install matplotlib numpy      # only for sweep.py
```

BM25 mode needs none of the heavy dependencies. The embedding model and the reranker are downloaded the first time you use them (about 90 MB each), then cached.

## How to run it

Ask one question:

```bash
python -m ask "How long do you keep my order with a bank transfer?"
```

Run `python -m velmora.ask` with no arguments for an interactive prompt.

Open the web interface (pick a question, change the parameters, run the whole evaluation):

```bash
python -m velmora.app --kb ai/kb 
# then open http://127.0.0.1:8000

```

Debug one question (scores, what counted as "covered", reranker order, conflicts):

```bash
python -m velmora.debug_q "My package has been stuck on the same status for a week." --kb ai/kb
```

## How to evaluate

```bash
# baseline: keyword search only
python -m velmora.evaluate --kb ai/kb --eval ai/eval.jsonl --method bm25

# best setup: embeddings + cross-encoder reranker
python -m velmora.evaluate --kb ai/kb --eval ai/eval.jsonl --method embeddings --rerank cross

# try several thresholds / the full tau x rho grid
python -m velmora.evaluate --kb ai/kb --eval ai/eval.jsonl --method embeddings --rerank cross --taus 0.4 0.45 0.5 0.55
python -m velmora.sweep    --kb ai/kb --eval ai/eval.jsonl --method embeddings --rerank cross
```

Only the question is given to the assistant. The expected answer, keywords and gold articles are used for scoring only.

## How to test

```bash
python -m pytest tests -v
```

The tests use a tiny three-article knowledge base and a fake assistant, so they finish in about a second and don't need any model. They check:

- recall@k, MRR and "all gold found" on small hand-made examples (including a miss),
- keyword matching (plurals, numbers, multi-word keywords, keywords that appear in the note),
- the five answer statuses, including the `null` fields used by unanswerable questions,
- the printed report, to make sure refusals are counted correctly.

## Results

40 questions: 35 answerable, 5 that the help center can't answer.

| | BM25 (baseline) | Embeddings + cross-encoder |
|---|---|---|
| Recall@1 | 0.80 | 0.89 |
| Recall@3 | 0.86 | 0.96 |
| Recall@5 | 0.87 | **1.00** |
| MRR | 0.84 | 0.95 |
| Good answers | 20 | **34** |
| Incomplete answers | 9 | 1 |
| False refusals | 6 | **0** |
| Correct refusals | 5 / 5 | 5 / 5 |
| False answers | 0 | 0 |
| Answer rate (answerable) | 0.83 | **1.00** |
| Cited a gold article | 0.90 | 0.97 |

So 39 of 40 questions are handled correctly in the best setup, up from 25 of 40.

**The one miss:** Q27, *"I can't remember how to get into my profile anymore."* The reranker put the account deletion article ahead of the password reset one, so the answer is confident and wrong ("Go to Settings, then Privacy, then Delete my account"). The right article (KB-20) was retrieved, just second. This is the kind of mistake I'd worry about most in real use.


## Why I built it this way

**Extractive answers, no generation.** The assistant picks the best sentences from the best articles and shows them as they are. It can't invent a delivery fee or a refund delay, which matters for a support tool. The price is that answers can read a bit stiff.

**Two-stage search.** BM25 is a keyword match, so it fails when the customer uses different words than the article ("get into my profile" vs "log in"). Sentence embeddings fix most of that. A cross-encoder then re-orders the shortlist by reading question and article together, which pushed Recall@1 from the embedding-only level up to 0.89.

**A threshold for "I don't know" (tau).** If no article scores at least `tau` (0.50 for embeddings), the assistant refuses. Too low and it answers things it shouldn't; too high and it refuses things it knows. The sweep script draws this trade-off as heatmaps so I could pick a stable zone instead of a lucky single value.

**Sentence ratio (rho).** Keeps up to 3 sentences per article, but only those scoring at least `rho` times the best one (0.4 for embeddings). It stops the answer from dumping the whole article.

**"Rescue" the reranker's first choice.** The reranker sometimes picks an article whose own embedding score is just below tau. If at least one article passes the threshold, I keep the reranker's top pick anyway. This fixed several false refusals. Set `RESCUE_BEST = False` in `answer.py` to turn it off.

**Splitting two-part questions.** "How fast is Express and what is the cut-off time?" is split on "and" and each half is searched separately, then the facts are merged. A guard stops it from splitting phrases like "terms and conditions".

**Duration normalising.** The articles say "7 days", customers say "a week". `normalize_question` rewrites it before searching.

**Old vs. new articles.** When two retrieved articles give different numbers for the same thing (30 days vs 14 days), the more recently updated one wins, and the answer includes a note saying the older article disagreed.

