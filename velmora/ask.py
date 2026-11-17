"""Ask the Velmora assistant a question.

Usage:
    python ask.py "How long do you keep my order with a bank transfer?"
    python ask.py            (interactive mode)
"""
import sys

from velmora.answer import Assistant, format_answer
from velmora.ingest import DEFAULT_MODEL, build_knowledge_base
from velmora.rerank import make_reranker

METHOD = "embeddings"  # "bm25" or "embeddings"
MODEL = DEFAULT_MODEL  # embedding model name
RERANK = "cross"       # None, "cross", "llm" or "semantic"


def main():
    kb = build_knowledge_base("kb", use_embeddings=(METHOD == "embeddings"), model_name=MODEL)  # phase 1: once
    assistant = Assistant(kb, method=METHOD, reranker=make_reranker(RERANK))  # phase 2: every question

    if len(sys.argv) > 1:
        print(format_answer(assistant.ask(" ".join(sys.argv[1:]))))
        return

    while True:
        question = input("\nQuestion (empty to quit): ").strip()
        if not question:
            break
        print(format_answer(assistant.ask(question)))


if __name__ == "__main__":
    main()