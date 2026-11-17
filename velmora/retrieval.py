"""Phase 2 - Step 1: find the best articles for a question."""
from velmora.text import normalize_question, tokenize


def encode_question(kb, question):
    """Question vector (embeddings mode only)."""
    return kb.model.encode([normalize_question(question)], normalize_embeddings=True)[0]


def score_articles(kb, question, method="bm25"):
    """Return one score per article ('bm25' or 'embeddings')."""
    if method == "embeddings":
        query = encode_question(kb, question)
        # An article scores like its best sentence (cosine similarity)
        return [float((kb.sentence_vectors[a.id] @ query).max()) for a in kb.articles]
    return kb.bm25.scores(tokenize(question))


def search(kb, question, top_n=5, method="bm25"):
    """Return the top_n (article, score) pairs, best first."""
    scores = score_articles(kb, question, method)
    ranked = sorted(zip(kb.articles, scores), key=lambda p: p[1], reverse=True)
    return [(article, score) for article, score in ranked[:top_n] if score > 0]
