"""Phase 1 - Knowledge base ingestion.

Load articles -> split sentences -> build BM25 index (and optional
sentence embeddings).
Run this once at startup, not for every question.
"""
import math
from collections import Counter
from dataclasses import dataclass

from velmora.data import load_articles
from velmora.text import search_text, split_sentences, tokenize


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class BM25:
    """Small BM25 index. Each article is one document."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.tf = [Counter(tokens) for tokens in docs_tokens]  # term counts per doc
        self.lengths = [len(tokens) for tokens in docs_tokens]
        self.avg_len = sum(self.lengths) / len(self.lengths)

        # Document frequency -> IDF (also used for the coverage check)
        n = len(docs_tokens)
        df = Counter(term for counts in self.tf for term in counts)
        self.idf = {
            term: math.log((n - d + 0.5) / (d + 0.5) + 1) for term, d in df.items()
        }
        # A term that is in no article is the rarest possible term
        self.unknown_idf = math.log((n + 0.5) / 0.5 + 1)

    def idf_of(self, term):
        """IDF weight of a term (unknown terms get the highest weight)."""
        return self.idf.get(term, self.unknown_idf)

    def scores(self, query_tokens):
        """Return one BM25 score per article, in the same order as the articles."""
        result = []
        for counts, length in zip(self.tf, self.lengths):
            score = 0.0
            for term in query_tokens:
                if term not in counts:
                    continue
                tf = counts[term]
                norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf[term] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result


@dataclass
class KnowledgeBase:
    articles: list                 # list of Article
    by_id: dict                    # "KB-01" -> Article
    sentences: dict                # "KB-01" -> list of sentences
    bm25: BM25 = None              # lexical index (article level)
    model: object = None           # embedding model (optional)
    sentence_vectors: dict = None  # "KB-01" -> array, one vector per sentence


def build_knowledge_base(
    kb_dir="kb",
    use_bm25=True,
    use_embeddings=False,
    model_name=DEFAULT_MODEL,
):
    """Load the articles and build the requested indexes."""
    articles = load_articles(kb_dir)

    kb = KnowledgeBase(
        articles=articles,
        by_id={a.id: a for a in articles},
        sentences={a.id: split_sentences(a.content) for a in articles},
    )

    if use_bm25:
        kb.bm25 = BM25([tokenize(search_text(a)) for a in articles])

    if use_embeddings:
        # Imported here so BM25 works without this dependency installed
        from sentence_transformers import SentenceTransformer

        kb.model = SentenceTransformer(model_name)
        kb.sentence_vectors = {}
        for a in articles:
            # The title is added to each sentence to keep its context
            texts = [f"{a.title}. {s}" for s in kb.sentences[a.id]]
            kb.sentence_vectors[a.id] = kb.model.encode(texts, normalize_embeddings=True)

    return kb
