"""Quick check: build the knowledge base and print a summary."""
from velmora.ingest import build_knowledge_base
from velmora.text import tokenize

kb = build_knowledge_base("kb")

print(f"{len(kb.articles)} articles loaded")
for a in kb.articles:
    print(f"- {a.id} | {a.category} | {a.updated} | {a.title}")

# Try a question
query = tokenize("Can I pay by bank transfer?")
scores = kb.bm25.scores(query)
best = max(range(len(scores)), key=scores.__getitem__)
print("\nTop article:", kb.articles[best].id, round(scores[best], 3))
