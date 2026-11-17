"""Phase 2 - Steps 2 to 4: coverage check, conflicts, extractive answer.

Multi-source version: decomposes multi-part questions (Q04, Q08), answers each
sub-question from its own best article(s), then merges the evidence.
"""
import re
from dataclasses import dataclass, field

from velmora.retrieval import encode_question, search
from velmora.text import keywords, normalize_question, search_text, tokenize

REFUSAL = (
    "I'm sorry, but I couldn't find enough information "
    "in the Velmora Help Center to answer this question."
)

DEFAULTS = {
    "bm25": {"tau": 0.5, "rho": 0.5},
    "embeddings": {"tau": 0.50, "rho": 0.4},
}
MAX_SENTENCES = 3        # at most 3 sentences per article
MIN_PART_KEYWORDS = 2    # a sub-question needs at least 2 keywords

RESCUE_BEST = True


@dataclass
class Answer:
    text: str
    sources: list
    answered: bool
    note: str = ""
    evidence: list = field(default_factory=list)  


@dataclass
class Candidate:
    article: object
    sentences: list


# ---------- step 3: query decomposition (simple rules) ----------

_SPLIT = re.compile(r"\s+(?:and|et)\s+|\s*;\s*|\?\s+", re.IGNORECASE)


def decompose(question):
    """Split 'A and B?' into sub-questions. Falls back to [question]."""
    parts = [p.strip(" ?.,") for p in _SPLIT.split(question) if p and p.strip(" ?.,")]
    if len(parts) < 2:
        return [question]
    if any(len(set(keywords(p))) < MIN_PART_KEYWORDS for p in parts):
        return [question]  # e.g. "terms and conditions" must not be split
    return [p + "?" for p in parts]


# ---------- helpers ----------

def coverage(bm25, terms, article):
    article_terms = set(tokenize(search_text(article)))
    total = sum(bm25.idf_of(t) for t in terms)
    found = sum(bm25.idf_of(t) for t in terms if t in article_terms)
    return found / total if total else 0.0


def sentence_score(bm25, terms, sentence):
    words = set(tokenize(sentence))
    return sum(bm25.idf_of(t) for t in terms if t in words)


_QUANTITY = re.compile(r"(\d+)\s+((?:business\s+|working\s+)?[a-z]+)")


def quantities(text):
    """'30 days' / '5 business days' -> {'day': {'30'}, 'business day': {'5'}}."""
    result = {}
    for number, unit in _QUANTITY.findall(text.lower()):
        result.setdefault(" ".join(tokenize(unit)), set()).add(number)
    return result


def words(text):
    """Topic words of a sentence: no digits, no stopwords (stopwords diluted the overlap)."""
    return {t for t in keywords(text) if not t.isdigit()}


def is_conflict(sentence_a, sentence_b, same_topic=0.2):
    """Same topic AND different values for the same unit (30 days vs 14 days)."""
    qty_a, qty_b = quantities(sentence_a), quantities(sentence_b)
    if not any(qty_a[u].isdisjoint(qty_b[u]) for u in qty_a if u in qty_b):
        return False
    words_a, words_b = words(sentence_a), words(sentence_b)
    if not words_a or not words_b:
        return False
    return len(words_a & words_b) / len(words_a | words_b) >= same_topic


def find_conflict(candidate_a, candidate_b):
    for sentence_a in candidate_a.sentences:
        for sentence_b in candidate_b.sentences:
            if is_conflict(sentence_a, sentence_b):
                return sentence_a, sentence_b
    return None


def resolve(candidates):
    """Pick the best candidate; every candidate that disagrees is compared by date
    and the most recent wins. Returns (chosen_candidate, sources, note)."""
    chosen, olds, notes = candidates[0], [], []
    for other in candidates[1:]:
        pair = find_conflict(chosen, other)
        if not pair:
            continue
        older, older_sentence = other, pair[1]
        if other.article.updated > chosen.article.updated:
            chosen, older, older_sentence = other, chosen, pair[0]
        olds.append(older.article)
        notes.append(
            f'Note: {older.article.id} (updated {older.article.updated}) '
            f'says: "{older_sentence}" '
            f"The most recent article, {chosen.article.id}, is used."
        )
    sources = [chosen.article] + [a for a in olds if a.id != chosen.article.id]
    return chosen, sources, "\n".join(notes)


# ---------- the assistant ----------

class Assistant:
    def __init__(self, kb, top_n=5, tau=None, rho=None, method="bm25", reranker=None):
        if method == "embeddings" and kb.sentence_vectors is None:
            raise ValueError("Build the knowledge base with use_embeddings=True")
        defaults = DEFAULTS[method]
        self.kb = kb
        self.top_n = top_n
        self.tau = defaults["tau"] if tau is None else tau
        self.rho = defaults["rho"] if rho is None else rho
        self.method = method
        self.reranker = reranker  

    def is_covered(self, terms, article, score):
        if self.method == "embeddings":
            return score >= self.tau
        return coverage(self.kb.bm25, terms, article) >= self.tau

    def sentence_scores(self, article, terms, query):
        if self.method == "embeddings":
            return (self.kb.sentence_vectors[article.id] @ query).tolist()
        return [sentence_score(self.kb.bm25, terms, s) for s in self.kb.sentences[article.id]]

    def pick_sentences(self, article, terms, query):
        sentences = self.kb.sentences[article.id]
        scores = self.sentence_scores(article, terms, query)
        ranked = sorted(zip(scores, range(len(sentences))), reverse=True)

        best_score = ranked[0][0] if ranked else 0
        if best_score <= 0:
            return None

        kept = [i for score, i in ranked[:MAX_SENTENCES] if score >= self.rho * best_score]
        kept.sort()
        return Candidate(article, [sentences[i] for i in kept])

    def rerank(self, question, candidates):
        """Put the most relevant candidate first (resolve() uses candidates[0])."""
        if not self.reranker or len(candidates) < 2:
            return candidates
        texts = [f"{c.article.title}. {c.article.content}" for c in candidates]
        scores = self.reranker.scores(normalize_question(question), texts)
        ranked = sorted(zip(scores, candidates), key=lambda p: p[0], reverse=True)
        return [c for _, c in ranked]

    def find_candidates(self, question):
        """Steps 1-2: top_n articles, keep covered ones with their sentences."""
        terms = set(keywords(question))
        query = encode_question(self.kb, question) if self.method == "embeddings" else None

        pool = []  # (candidate, covered)
        for article, score in search(self.kb, question, self.top_n, self.method):
            candidate = self.pick_sentences(article, terms, query)
            if candidate:
                pool.append((candidate, self.is_covered(terms, article, score)))

        if not any(covered for _, covered in pool):
            return []  # nothing covered -> refusal (unchanged behaviour)

        if not (RESCUE_BEST and self.reranker):
            return self.rerank(question, [c for c, covered in pool if covered])

        # The reranker sees every retrieved article, not only the covered ones
        ranked = self.rerank(question, [c for c, _ in pool])
        covered_ids = {c.article.id for c, covered in pool if covered}
        best = ranked[0]
        return [c for c in ranked if c.article.id in covered_ids or c is best]

    def ask(self, question):
        evidence, sources, notes, missing = [], [], [], []
        seen_sentences = set()

        # Step 3: one retrieval per sub-question
        for part in decompose(question):
            candidates = self.find_candidates(part)
            if not candidates:
                missing.append(part)
                continue
            chosen, part_sources, note = resolve(candidates)  # Step 4
            for sentence in chosen.sentences:
                if sentence not in seen_sentences:  # no duplicates
                    seen_sentences.add(sentence)
                    evidence.append({"article_id": chosen.article.id, "text": sentence})
            for article in part_sources:
                if article.id not in {s.id for s in sources}:
                    sources.append(article)
            if note and note not in notes:
                notes.append(note)

        if not evidence:
            return Answer(REFUSAL, [], answered=False)

        for part in missing:  # partial answer: say what is missing
            notes.append(f'Note: no information found for "{part}".')

        # Step 5: extractive answer, facts combined in order
        text = " ".join(e["text"] for e in evidence)
        return Answer(text, sources, True, "\n".join(notes), evidence)


def format_answer(answer):
    lines = [answer.text]
    if answer.note:
        lines.append(f"\n{answer.note}")
    if answer.sources:
        lines.append("\nSources:")
        for a in answer.sources:
            lines.append(f"- {a.id} · {a.title} · Updated: {a.updated}")
    return "\n".join(lines)