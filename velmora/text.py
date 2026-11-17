"""Text cleaning and tokenization (used for articles AND questions)."""
import re


def simple_stem(token):
    """Very light stemming: takes -> take, days -> day (same for questions)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


# Question words with no topic meaning. Only used to pick the question keywords.
_STOPWORDS = {
    "a", "an", "and", "any", "are", "at", "be", "by", "can", "could", "do",
    "does", "for", "have", "how", "i", "if", "in", "is", "it", "long", "many",
    "me", "much", "my", "of", "on", "or", "should", "that", "the", "there",
    "this", "to", "we", "what", "when", "where", "which", "will", "with",
    "would", "you", "your",
}
STOPWORDS = {simple_stem(w) for w in _STOPWORDS}  # same stemming as the tokens


def clean_text(text):
    """Normalize text for search. The original article text is not changed."""
    text = text.lower()
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"[#*_`>|~]", " ", text)                # markdown symbols
    text = text.replace("’", "'").replace("'", "")        # don't -> dont
    return re.sub(r"\s+", " ", text).strip()              # collapse spaces


def tokenize(text):
    """Return a list of tokens. Numbers and payment names are kept."""
    return [simple_stem(t) for t in re.findall(r"[a-z0-9]+", clean_text(text))]


def keywords(text):
    """Tokens without stopwords (the important terms of a question)."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def search_text(article):
    """Text used for indexing: title + content (the title helps matching)."""
    return f"{article.title}. {article.content}"


def split_sentences(text):
    """Split an article body into sentences (heading lines are skipped)."""
    sentences = []
    for line in text.splitlines():
        if not line.startswith("#"):
            sentences += re.split(r"(?<=[.!?])\s+", line.strip())
    return [s for s in sentences if s]


_WEEKS = re.compile(r"\b(a|an|one|1|two|2|three|3|four|4)\s+weeks?\b", re.IGNORECASE)
_WORD_NUM = {"a": 1, "an": 1, "one": 1, "1": 1, "two": 2, "2": 2,
             "three": 3, "3": 3, "four": 4, "4": 4}


def normalize_question(question):
    """'a week' -> '7 days' (articles express durations in days)."""
    return _WEEKS.sub(lambda m: f"{_WORD_NUM[m.group(1).lower()] * 7} days", question)
