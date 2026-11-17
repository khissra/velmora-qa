# Velmora help-center assistant and evaluation harness

A small question-answering assistant for the Velmora help center (30 articles), and a harness that tells us
**how well it answers and when it should say "I don't know"**.

- The assistant searches the articles and **copies** the best sentence(s) word for word, with the article ids.
  Nothing is generated, so a sentence cannot be invented. If the help center does not cover the question, it refuses.
- The harness runs all 40 questions with one command, prints a summary, and writes one line per question.
- No API key, no account, no language model. Local embeddings are used for the final configuration (see "Network").

## Results at a glance

Two configurations, same 40 questions (35 answerable, 5 not), same `tau = 0.5`, nothing tuned on the questions
(see "How the 40 questions were used").

| | **BM25 (baseline)** | **Hybrid + dense + expand (final)** |
|---|---|---|
| recall@1 | 28/35 = 0.80 | 31/35 = 0.89 |
| recall@3 | 32/35 = 0.91 | 35/35 = 1.00 |
| recall@5 | 32/35 = 0.91 | 35/35 = 1.00 |
| Correct answers (keywords AND cited source) | 25/35 = 0.71 | **33/35 = 0.94** |
| Refused by mistake (answerable) | 7/35 = 0.20 | **1/35 = 0.03** |
| Correctly refused (unanswerable) | 5/5 | 5/5 |
| Invented an answer (unanswerable) | 0/5 | 0/5 |
| All 40 behaved correctly | 30/40 = 0.75 | 38/40 = 0.95 |

95% Wilson intervals for the correct answers: BM25 [0.55, 0.84], final [0.81, 0.98]. For the refusals on the
unanswerable questions: [0.57, 1.00] in both cases (only 5 questions).
Stricter retrieval (all gold articles in the top-k): BM25 0.77 / 0.89 / 0.89, final 0.89 / 0.94 / 0.97 for k = 1 / 3 / 5.

**Read these numbers with care.** The final configuration was built after reading the BM25 failures, so its
score is **optimistic**, not a held-out score. See "How the 40 questions were used" and "Limits of the evaluation".

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # sentence-transformers (only for the final configuration) and pytest
```

```bash
# FINAL configuration: one command, prints the summary, writes results_hybrid_dense_expand.jsonl
python -m velmora.evaluate --kb ai/kb --data ai/eval.jsonl --out results_hybrid_dense_expand.jsonl \
       --method hybrid --dense --expand --tau 0.5 --seed 42 --sweep

# BASELINE (BM25 only, standard library only): writes results.jsonl
python -m velmora.evaluate --kb ai/kb --data ai/eval.jsonl --out results.jsonl \
       --method bm25 --tau 0.5 --seed 42 --sweep

# All 5 configurations side by side + paired differences (fixed / broken questions)
python -m velmora.evaluate --kb ai/kb --data ai/eval.jsonl --out results_compare.jsonl \
       --tau 0.5 --seed 42 --compare

# Ask one question
python -m velmora.ask --method hybrid --dense --expand --tau 0.5 "How do I get my money back?"
```

Notes:

- The system is deterministic: no randomness anywhere. The seed is set anyway for any future random part.
- `kb/` and `eval.jsonl` are **only read**. Their SHA-256 hashes are printed before and after the run and compared
  (`unchanged during run: True`; exit code 2 if not).
- `evaluate.py` uses `tau = 0.5` by default and `ask.py` uses `0.3`. All numbers here use 0.5, so pass `--tau 0.5` to `ask.py`.
- **Network:** BM25 needs nothing. The final configuration needs the embedding model `all-MiniLM-L6-v2`, which is
  **downloaded once** on the first run (from Hugging Face) and then works offline from the local cache.
- Versions are not pinned in `requirements.txt`. For exact reproduction of the embeddings, run `pip freeze` in your
  environment and keep that file.

## How to test

```bash
python -m pytest tests -q
```

14 tests, about 0.1 second. They use BM25 only (no model, no network). They cover:

- `recall@k` on a hand-made ranking (`any` / `all` modes, error on empty gold);
- the keyword judge (whole-token match: "7 days" must not match "17 days", "4.90" must not match "14.90");
- counting of correct refusals, wrongful refusals and invented answers;
- the Wilson interval and the paired comparison (`fixed` / `broken`);
- data loading errors (missing header, bad date, inconsistent eval line);
- tokenizer and sentence splitter (numbers such as `7.90` are kept);
- BM25 (relevant article first, deterministic, unknown words score 0);
- conflict detection (different durations or amounts);
- the assistant end to end: cited extractive answer, refusal of an out-of-scope question, per-question outcome.

The hybrid / dense / expand configuration is checked by the evaluation itself, not by unit tests.

## What the assistant does

1. **Expand** (optional, `--expand`): hand-written rules add the help center's wording to the question
   ("parcel" <-> "package", "a week" <-> "7 days", "I can't get into my profile" -> "login password reset").
2. **Rank articles**: BM25 (`k1 = 1.5`, `b = 0.75`, textbook values) or, with `--method hybrid`, BM25 and embeddings
   merged with Reciprocal Rank Fusion (`k = 60`).
3. **Score sentences** of the top 5 articles: word overlap (idf), a bonus when the sentence holds the value the question
   expects (a duration for "how long", an amount for "how much", a list for "which methods"), a title bonus, and
   with `--dense` a semantic similarity between the question and each sentence ("money back" ~ "refund").
   The article of the best sentence becomes the main article.
4. **Refuse**: coverage = share of the question's words (weighted by idf) found in the main article.
   Below `tau` the answer is "I don't know", with **no text and no citation**.
5. **Answer**: the best sentence, copied word for word, plus a neighbour sentence only if it also holds the expected value.
6. **Conflicts and dates**: if two articles give different durations or amounts for the same thing, both are shown and the
   conflict is announced with the update dates. An article more than 365 days older than another is treated as outdated.

## Results in detail

### Comparison of the 5 configurations (`--compare`, same questions, tau = 0.5)

| Configuration | rec@1 | rec@3 | correct | refused by mistake | refused ok | invented |
|---|---|---|---|---|---|---|
| bm25 | 28/35 | 32/35 | 25/35 | 7/35 | 5/5 | 0/5 |
| bm25 + expand | 29/35 | 32/35 | 25/35 | 7/35 | 5/5 | 0/5 |
| hybrid | 31/35 | 33/35 | 25/35 | 7/35 | 5/5 | 0/5 |
| hybrid + dense | 31/35 | 33/35 | 29/35 | 4/35 | 5/5 | 0/5 |
| **hybrid + dense + expand** | 31/35 | 35/35 | **33/35** | **1/35** | 5/5 | 0/5 |

With 35 questions the confidence intervals of these configurations overlap, so the **paired differences** are the honest
view (questions fixed or broken compared with BM25):

| Configuration | Fixed | Broken |
|---|---|---|
| bm25 + expand | Q02, Q22 | Q20, Q23 |
| hybrid | none | none |
| hybrid + dense | Q08, Q10, Q22, Q39, Q40 | Q14 |
| hybrid + dense + expand | Q02, Q05, Q08, Q10, Q15, Q17, Q22, Q27, Q39, Q40 | Q14, Q20 |

What this shows:

- **Hybrid alone does not change the number of correct answers** (25 -> 25), even if recall@1 goes from 28 to 31.
  Better ranking is not enough: the loss was in the refusal step and in sentence choice.
- **Dense sentence scoring is the biggest single gain** (+4 correct): it fixes Q08, Q10, Q22, Q39 and Q40.
- **Expansion alone does nothing** (25 -> 25, two fixed and two broken). It helps only **together** with dense
  (Q05, Q15, Q17, Q27 are fixed in the full configuration, not before).
- **Net effect of the final configuration: +10 questions fixed, 2 broken.** Both broken ones (Q14, Q20) are analysed below.
- **No question without an answer got an answer in any configuration.**

### Refusal threshold (final configuration; sensitivity analysis, not tuning)

| tau | correct refusals | wrongful refusals | correct answers |
|---|---|---|---|
| 0.10 | 1/5 | 0/35 | 33/35 |
| 0.20 | 2/5 | 0/35 | 33/35 |
| 0.30 | 5/5 | 0/35 | 33/35 |
| 0.40 | 5/5 | 1/35 | 33/35 |
| **0.50 (used)** | 5/5 | 1/35 | 33/35 |
| 0.60 | 5/5 | 4/35 | 30/35 |
| 0.70 | 5/5 | 10/35 | 25/35 |
| 0.80 | 5/5 | 16/35 | 19/35 |
| 0.90 | 5/5 | 17/35 | 18/35 |

The curve is flat between 0.30 and 0.50 and drops quickly after 0.60. Below 0.30 the assistant starts to answer
unanswerable questions. **The margin is thin**, though: three correct answers pass with coverage 0.514 (Q17),
0.516 (Q10) and 0.573 (Q15). At tau = 0.6 four answerable questions are refused.
For BM25 the same table is printed by `--sweep` (e.g. tau 0.5 -> 7 wrongful refusals, tau 0.3 -> 4).

## Failure analysis

See `FAILURES.md`: all 10 failures of the BM25 baseline grouped by cause (vocabulary mismatch, coverage too low,
wrong article or sentence, value type not recognized), and the 2 remaining failures plus the weak "correct" answers of the
final configuration. Short version of what remains:

- **Q14** (wrong answer): the right article is first, but a duration sentence from another article (delivery time to a
  pickup point) wins over "held for 10 days".
- **Q20** (wrongful refusal): the expansion adds words ("send back", "give back", "take back") that are not in the
  article, which lowers the coverage to 0.36.

## How the 40 questions were used (honest note)

- **No numeric parameter was tuned on `eval.jsonl`.** BM25 values, stop words, bonuses (1.5, 1.0, 0.5), the
  365-day window and `tau = 0.5` were fixed before the runs. The tau table is a sensitivity analysis only.
- **But the code was designed while looking at the questions.** The conflict rule was written after reading Q04
  (KB-12 vs KB-13); the type bonus, title bonus and amount conflicts after the error analysis of Q02, Q03, Q07; more rules
  after a second analysis of the same questions.
- **The expansion rules are not held-out either.** Even if they are written as generic shop rules, I have seen the
  questions, and several rules match the wording of failing questions (the "get into my" login rule and Q27,
  "by mistake" and Q15, "same status" and Q05, the country rule and Q17). The gain on those four questions should
  not be counted as independent evidence.
- **Consequence:** 33/35 is an upper estimate of the real performance on new questions. The way to measure the real
  value is to write new questions **before** running the assistant on them (planned, see the design note).
- With 5 unanswerable questions, "5/5 refused" has a 95% interval of [57%, 100%]. It does not prove safety.

## Limits of the evaluation

- **The judge is keyword-based.** It accepts an answer that contains the expected words even if it is flawed. Examples
  counted as correct in the final run: Q04 starts with the **refund** sentence (5 business days) before the return window
  (30 days) and reports it as a conflicting value; Q10 adds an unrelated pickup-point sentence; Q17 starts with a sentence
  about customs fees. It can also reject a correct answer written differently. It has not been checked against a manual
  review of all answers.
- Q04 and Q08 have two gold articles; "correct" needs **one** of them (`recall_all@k` shows the stricter view).
- Data anomalies in `kb/` or `eval.jsonl` are **reported, not fixed**: `python -m velmora.data --kb ai/kb --eval ai/eval.jsonl`.
- Small sample: one question changes a rate by about 3 points.

## Use of AI tools

An AI assistant (Claude) helped with the code review, the tests and the documentation. I ran every command myself
and can explain and change the code. The assistant itself uses **no language model**.

## Project layout

```
velmora/
  data.py        load kb/ and eval.jsonl (read-only), hashes, audit
  text.py        tokenizer, light stemmer, sentence splitter
  retrieval.py   BM25, TF-IDF, vector, hybrid (RRF), sentence-level dense retriever
  patterns.py    all regular expressions (value types, durations, intents)
  expand.py      concept expansion of the question (optional)
  answer.py      extractive answer, refusal, conflicts and dates
  metrics.py     recall@k, keyword judge, Wilson interval, paired comparison
  evaluate.py    the one-command harness (--sweep, --compare)
  ask.py         ask one question
tests/
  test_bm25_pipeline.py
FAILURES.md
requirements.txt
```

`velmora/dense.py` is not used by the pipeline (the sentence-level retriever is in `retrieval.py`); it can be deleted.

## Next steps

Described in the design note: new held-out questions, a manual check of the judge, a safer refusal rule
(do not let expansion lower the coverage), and what to do about invented answers, old or conflicting articles, and
questions with no answer.
