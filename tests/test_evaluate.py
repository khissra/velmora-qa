"""Tests for velmora/evaluate.py.

They use a tiny hand-made knowledge base and a fake assistant, so they run in
under a second and need no model download (BM25 only).

Run from the folder that contains the velmora package:
    python -m pytest tests -v
"""
from velmora.answer import Answer
from velmora.data import Article
from velmora.evaluate import (
    evaluate_question,
    get_status,
    keywords_found,
    mean,
    print_report,
)
from velmora.ingest import BM25, KnowledgeBase
from velmora.text import search_text, split_sentences, tokenize


# ---------- tiny fixtures ----------

def make_kb():
    articles = [
        Article("KB-01", "Delivery", "shipping", "2026-01-01", "Standard delivery takes 3 days."),
        Article("KB-02", "Returns", "returns", "2026-01-01", "Returns are accepted within 30 days."),
        Article("KB-03", "Gift cards", "payment", "2026-01-01", "A gift card is valid for one year."),
    ]
    return KnowledgeBase(
        articles=articles,
        by_id={a.id: a for a in articles},
        sentences={a.id: split_sentences(a.content) for a in articles},
        bm25=BM25([tokenize(search_text(a)) for a in articles]),
    )


class FakeAssistant:
    """Returns a fixed Answer, whatever the question is."""

    method = "bm25"

    def __init__(self, kb, answer):
        self.kb = kb
        self._answer = answer

    def ask(self, question):
        return self._answer


def answered(text, kb, ids=(), note=""):
    return Answer(text, [kb.by_id[i] for i in ids], answered=True, note=note)


def refused():
    return Answer("I'm sorry, I couldn't find enough information.", [], answered=False)


# ---------- 1. recall@k, MRR, all_gold ----------

def test_recall_is_one_when_the_gold_article_is_first():
    kb = make_kb()
    item = {"id": "T1", "question": "How long does standard delivery take?",
            "answerable": True, "source_articles": ["KB-01"], "answer_keywords": ["3 days"]}
    row = evaluate_question(FakeAssistant(kb, answered("Standard delivery takes 3 days.", kb, ["KB-01"])), item)

    assert row["retrieved"][0] == "KB-01"
    assert row["recall"][1] == 1.0
    assert row["rr"] == 1.0
    assert row["all_gold"] is True


def test_recall_with_two_gold_articles_is_a_fraction():
    kb = make_kb()
    # "delivery returns" matches KB-01 and KB-02, so top-1 can only hold one of the two
    item = {"id": "T2", "question": "delivery returns", "answerable": True,
            "source_articles": ["KB-01", "KB-02"], "answer_keywords": []}
    row = evaluate_question(FakeAssistant(kb, answered("x", kb, ["KB-01"])), item)

    assert row["recall"][1] == 0.5
    assert row["recall"][3] == 1.0
    assert row["all_gold"] is True


def test_recall_is_zero_when_the_gold_article_is_not_retrieved():
    kb = make_kb()
    item = {"id": "T3", "question": "delivery", "answerable": True,
            "source_articles": ["KB-03"], "answer_keywords": []}
    row = evaluate_question(FakeAssistant(kb, refused()), item)

    assert row["recall"][5] == 0.0
    assert row["rr"] == 0.0
    assert row["all_gold"] is False


# ---------- 2. keyword matching ----------

def test_keywords_found_handles_plurals_numbers_and_case():
    text = "You have 30 Days to return the item."
    assert keywords_found(text, ["30 days", "14 days"]) == ["30 days"]


def test_keywords_found_needs_every_word_of_a_multiword_keyword():
    assert keywords_found("Refunds take 5 days.", ["5 business days"]) == []
    assert keywords_found("Refunds take 5 business days.", ["5 business days"]) == ["5 business days"]


def test_keywords_in_the_note_count_as_shown_to_the_user():
    kb = make_kb()
    item = {"id": "T4", "question": "delivery", "answerable": True,
            "source_articles": ["KB-01"], "answer_keywords": ["3 days", "old article"]}
    answer = answered("Standard delivery takes 3 days.", kb, ["KB-01"],
                      note="Note: this replaces an old article.")
    row = evaluate_question(FakeAssistant(kb, answer), item)

    assert row["missing"] == []
    assert row["status"] == "good_answer"


# ---------- 3. status: refusals and answers ----------

def test_get_status_covers_the_five_cases():
    assert get_status(answerable=True, answered=True, keyword_score=1.0) == "good_answer"
    assert get_status(answerable=True, answered=True, keyword_score=0.5) == "incomplete_answer"
    assert get_status(answerable=True, answered=False, keyword_score=1.0) == "false_refusal"
    assert get_status(answerable=False, answered=False, keyword_score=1.0) == "correct_refusal"
    assert get_status(answerable=False, answered=True, keyword_score=1.0) == "false_answer"


def test_unanswerable_question_with_null_fields():
    """eval.jsonl uses null for source_articles / answer_keywords on unanswerable questions."""
    kb = make_kb()
    item = {"id": "T5", "question": "Do you have a shop in Lyon?", "answerable": False,
            "source_articles": None, "answer_keywords": None}

    refusal_row = evaluate_question(FakeAssistant(kb, refused()), item)
    assert refusal_row["status"] == "correct_refusal"
    assert "recall" not in refusal_row  # no gold -> no retrieval score

    wrong_row = evaluate_question(FakeAssistant(kb, answered("Yes, in Lyon.", kb)), item)
    assert wrong_row["status"] == "false_answer"


# ---------- 4. the report ----------

def test_report_counts_refusals(capsys):
    kb = make_kb()
    good = {"id": "A1", "question": "How long does standard delivery take?", "answerable": True,
            "source_articles": ["KB-01"], "answer_keywords": ["3 days"]}
    false_refusal = {"id": "A2", "question": "How many days to return?", "answerable": True,
                     "source_articles": ["KB-02"], "answer_keywords": ["30 days"]}
    no_answer = {"id": "U1", "question": "Do you sell cars?", "answerable": False,
                 "source_articles": None, "answer_keywords": None}

    rows = [
        evaluate_question(FakeAssistant(kb, answered("Standard delivery takes 3 days.", kb, ["KB-01"])), good),
        evaluate_question(FakeAssistant(kb, refused()), false_refusal),
        evaluate_question(FakeAssistant(kb, refused()), no_answer),
    ]
    print_report(rows)
    out = capsys.readouterr().out

    assert "Questions: 3 (2 answerable, 1 unanswerable)" in out
    assert "Answer rate (answerable):      0.50" in out
    assert "Refusal rate (unanswerable):   1.00" in out
    assert "false_refusal      1" in out


def test_mean_of_an_empty_list_is_zero():
    assert mean([]) == 0.0
    assert mean([True, False, True, True]) == 0.75
