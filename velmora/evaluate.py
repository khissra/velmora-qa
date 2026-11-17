"""Phase 3 - Evaluate the assistant on eval.jsonl.

Only the question is sent to the assistant. The expected answer, keywords
and source articles are used for scoring only (no data leakage).
"""
import argparse
import json
from collections import Counter

from velmora.answer import Assistant
from velmora.ingest import DEFAULT_MODEL, build_knowledge_base
from velmora.rerank import make_reranker
from velmora.retrieval import search
from velmora.text import tokenize

KS = (1, 3, 5)  # Recall@K values; the largest one is also "top-5"
CORRECT_STATUSES = ("good_answer", "correct_refusal")


def load_questions(path):
    """Read one JSON question per line."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keywords_found(text, keywords):
    """Keywords present in the text (same tokenization as the search)."""
    words = set(tokenize(text))
    return [k for k in keywords if all(t in words for t in tokenize(k))]


def get_status(answerable, answered, keyword_score):
    """Classify the behaviour of the assistant for one question."""
    if answerable:
        if not answered:
            return "false_refusal"        # refused but the answer exists
        return "good_answer" if keyword_score == 1 else "incomplete_answer"
    return "false_answer" if answered else "correct_refusal"


def evaluate_question(assistant, item):
    """Run one question and compute its scores."""
    question = item["question"]
    gold = item.get("source_articles") or []  # null -> empty list
    keywords = item.get("answer_keywords") or []

    results = search(assistant.kb, question, max(KS), assistant.method)
    retrieved = [a.id for a, _ in results]
    answer = assistant.ask(question)

    row = {
        "id": item["id"],
        "question": question,
        "answerable": item["answerable"],
        "answered": answer.answered,
        "retrieved": retrieved,
        "gold": gold,
        "answer": answer.text,
        "cited": [a.id for a in answer.sources],
        "note": answer.note,
        "top_score": results[0][1] if results else 0.0,  # helps to choose tau
    }

    # Retrieval scores (only when the question has gold articles)
    if gold:
        row["recall"] = {k: len(set(retrieved[:k]) & set(gold)) / len(gold) for k in KS}
        ranks = [retrieved.index(g) + 1 for g in gold if g in retrieved]
        row["rr"] = 1 / min(ranks) if ranks else 0.0
        row["all_gold"] = set(gold) <= set(retrieved)
        row["cited_gold"] = bool(set(row["cited"]) & set(gold))

    # Answer quality (keywords found in the answer text)
    # The user sees the text AND the note (conflicts are shown there)
    shown = f"{answer.text} {answer.note}"
    found = keywords_found(shown, keywords)
    row["missing"] = [k for k in keywords if k not in found]
    row["keyword_score"] = len(found) / len(keywords) if keywords else 1.0

    row["status"] = get_status(item["answerable"], answer.answered, row["keyword_score"])
    return row


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_score_table(title, rows):
    """Print one table of top scores, sorted from highest to lowest."""
    print(f"\n{title} ({len(rows)})")
    if not rows:
        print("  (none)")
        return
    print(f"  {'id':<5} {'top score':<10} {'status':<18} {'answerable':<11} question")
    for r in sorted(rows, key=lambda r: r["top_score"], reverse=True):
        print(
            f"  {r['id']:<5} {r['top_score']:<10.2f} {r['status']:<18} "
            f"{str(r['answerable']):<11} {r['question'][:60]}"
        )


def print_scores(rows):
    """Top retrieval score per question, split into correct / incorrect."""
    correct = [r for r in rows if r["status"] in CORRECT_STATUSES]
    incorrect = [r for r in rows if r["status"] not in CORRECT_STATUSES]

    print("\n== Top score par question ==")
    print_score_table("Réponses correctes (good_answer + correct_refusal)", correct)
    print_score_table(
        "Réponses incorrectes (incomplete_answer + false_refusal + false_answer)",
        incorrect,
    )

    if correct:
        scores = [r["top_score"] for r in correct]
        print(f"\nCorrectes   : min {min(scores):.2f}  mean {mean(scores):.2f}  max {max(scores):.2f}")
    if incorrect:
        scores = [r["top_score"] for r in incorrect]
        print(f"Incorrectes : min {min(scores):.2f}  mean {mean(scores):.2f}  max {max(scores):.2f}")


def print_report(rows):
    """Print the summary and the list of problems."""
    with_gold = [r for r in rows if "recall" in r]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    count = lambda status: sum(r["status"] == status for r in rows)

    print(f"Questions: {len(rows)} ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")

    print("\n== Retrieval ==")
    for k in KS:
        print(f"Recall@{k}:            {mean([r['recall'][k] for r in with_gold]):.2f}")
    print(f"MRR:                 {mean([r['rr'] for r in with_gold]):.2f}")
    print(f"All gold in top-{max(KS)}:  {mean([r['all_gold'] for r in with_gold]):.2f}")

    print("\n== Answers and refusals ==")
    for status in ("good_answer", "incomplete_answer", "false_refusal",
                   "correct_refusal", "false_answer"):
        print(f"{status:18} {count(status)}")
    print(f"Answer rate (answerable):      {mean([r['answered'] for r in answerable]):.2f}")
    print(f"Refusal rate (unanswerable):   {mean([not r['answered'] for r in unanswerable]):.2f}")
    answered = [r for r in with_gold if r["answered"]]
    print(f"Cited a gold article:          {mean([r['cited_gold'] for r in answered]):.2f}")
    print(f"Mean keyword score (answered): {mean([r['keyword_score'] for r in rows if r['answered'] and r['answerable']]):.2f}")

    # Top retrieval score: use it to choose tau (answerable should be higher)
    if answerable and unanswerable:
        a_scores = [r["top_score"] for r in answerable]
        u_scores = [r["top_score"] for r in unanswerable]
        print(f"Top score, answerable:   min {min(a_scores):.2f}  mean {mean(a_scores):.2f}")
        print(f"Top score, unanswerable: max {max(u_scores):.2f}  mean {mean(u_scores):.2f}")

    print_scores(rows)

    print("\n== Problems to check ==")
    for r in rows:
        if r["status"] in CORRECT_STATUSES:
            continue
        print(f"\n[{r['status']}] {r['id']}: {r['question']}")
        print(f"  gold: {r['gold']}  retrieved: {r['retrieved']}  cited: {r['cited']}")
        print(f"  top score: {r['top_score']:.2f}  missing keywords: {r['missing']}")
        print(f"  answer: {r['answer'][:150]}")


def print_sweep(kb, questions, args, reranker):
    """Try several tau values and print one line per value."""
    names = ["good_answer", "incomplete_answer", "false_refusal",
             "correct_refusal", "false_answer"]
    print("tau    " + "  ".join(names))
    for tau in args.taus:
        assistant = Assistant(kb, tau=tau, rho=args.rho, method=args.method, reranker=reranker)
        rows = [evaluate_question(assistant, item) for item in questions]
        counts = Counter(r["status"] for r in rows)
        print(f"{tau:<6} " + "  ".join(f"{counts[n]:<{len(n)}}" for n in names))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Velmora assistant")
    parser.add_argument("--kb", default="kb")
    parser.add_argument("--eval", default="eval.jsonl")
    parser.add_argument("--method", default="bm25", choices=["bm25", "embeddings"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model name")
    parser.add_argument("--rerank", choices=["cross", "llm", "semantic"],
                        help="optional reranker (default: none)")
    parser.add_argument("--tau", type=float, help="threshold (default depends on method)")
    parser.add_argument("--rho", type=float, help="sentence ratio (default depends on method)")
    parser.add_argument("--taus", type=float, nargs="+",
                        help="try several tau values, e.g. --taus 0.35 0.4 0.45")
    parser.add_argument("--out", help="optional JSON file for per-question results")
    args = parser.parse_args()

    kb = build_knowledge_base(
        args.kb, use_embeddings=(args.method == "embeddings"), model_name=args.model
    )
    reranker = make_reranker(args.rerank)  # None if --rerank is not set
    assistant = Assistant(kb, tau=args.tau, rho=args.rho, method=args.method, reranker=reranker)

    questions = load_questions(args.eval)
    if args.taus:
        print_sweep(kb, questions, args, reranker)
        return

    rows = [evaluate_question(assistant, item) for item in questions]
    print_report(rows)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()