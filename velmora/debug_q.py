"""Debug one question: scores, coverage, reranker order, conflicts.

Usage: python -m velmora.debug_q "My package has been stuck on the same status for a week." --kb ai/kb
(or run it from the same place as ask.py: python debug_q.py "..." --kb ai/kb)
"""
import argparse

from velmora.answer import Assistant, find_conflict
from velmora.ingest import DEFAULT_MODEL, build_knowledge_base
from velmora.rerank import make_reranker
from velmora.retrieval import search
from velmora.text import keywords

p = argparse.ArgumentParser()
p.add_argument("question")
p.add_argument("--kb", default="kb")
args = p.parse_args()

kb = build_knowledge_base(args.kb, use_embeddings=True, model_name=DEFAULT_MODEL)
reranker = make_reranker("cross")
a = Assistant(kb, method="embeddings", reranker=reranker)

terms = set(keywords(args.question))
print(f"tau = {a.tau}\n")
print("== Retrieval (embedding score, covered?) ==")
for art, score in search(kb, args.question, a.top_n, "embeddings"):
    print(f"  {art.id}  {score:.2f}  covered={a.is_covered(terms, art, score)}  {art.title}")

cands = a.find_candidates(args.question)
print("\n== Candidates after rerank (first = used) ==")
for c in cands:
    print(f"  {c.article.id}: {c.sentences}")

print("\n== Conflicts ==")
for i, x in enumerate(cands):
    for y in cands[i + 1:]:
        print(f"  {x.article.id} vs {y.article.id}: {find_conflict(x, y)}")
