"""Web interface for the Velmora assistant (Python standard library only).

The page lets the user:
  - pick any question of eval.jsonl (or type their own),
  - change the parameters (method, reranker, tau, rho, top_n),
  - run the full evaluation with those parameters and read the results.

Usage (from the folder that contains the velmora package, like ask.py):
    python app.py --kb ai/kb
    python app.py --kb ai/kb --eval ai/eval.jsonl --port 8000
    python app.py --kb ai/kb --offline          # BM25 only, no model download

Then open http://127.0.0.1:8000

Endpoints:
    GET  /                    the page (index.html, next to this file)
    GET  /api/config          available methods, rerankers, defaults
    GET  /api/questions       the questions of eval.jsonl (id, question, answerable)
    GET  /api/article/<id>    one article (to read a cited source)
    POST /api/ask             {"question": "...", "params": {...}}
    POST /api/evaluate        {"params": {...}} -> summary + one row per question
"""
import argparse
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from velmora.answer import DEFAULTS, Assistant
from velmora.evaluate import KS, evaluate_question, load_questions, mean
from velmora.ingest import DEFAULT_MODEL, build_knowledge_base
from velmora.rerank import make_reranker
from velmora.retrieval import search
from velmora.text import keywords

INDEX = Path(__file__).with_name("index.html")
MAX_QUESTION_CHARS = 500
RERANKERS = ["none", "cross", "semantic", "llm"]
DEFAULT_TOP_N = 5
LOCK = threading.Lock()  # models are not guaranteed to be thread-safe: one job at a time
STATE = {}


# ---------- parameters ----------

def parse_params(payload):
    """Validate the parameters sent by the page. Raises ValueError with a readable message."""
    raw = payload.get("params") or {}
    if not isinstance(raw, dict):
        raise ValueError("'params' must be an object")
    config = STATE["config"]

    method = raw.get("method", config["method"])
    if method not in config["methods"]:
        raise ValueError(f"Method '{method}' is not available (available: {config['methods']})")

    rerank = raw.get("rerank", config["reranker"])
    if rerank not in RERANKERS:
        raise ValueError(f"Unknown reranker '{rerank}'")

    def number(name, low, high, cast):
        value = raw.get(name)
        if value in (None, ""):
            return None
        try:
            value = cast(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number")
        if not low <= value <= high:
            raise ValueError(f"'{name}' must be between {low} and {high}")
        return value

    return {
        "method": method,
        "rerank": rerank,
        "tau": number("tau", 0.0, 1.0, float),
        "rho": number("rho", 0.0, 1.0, float),
        "top_n": number("top_n", 1, 10, int) or DEFAULT_TOP_N,
    }


def get_reranker(name):
    """Load a reranker once, the first time it is needed (call under LOCK)."""
    if name == "none":
        return None
    if name not in STATE["rerankers"]:
        STATE["rerankers"][name] = make_reranker(name)
    return STATE["rerankers"][name]


def build_assistant(params):
    return Assistant(
        STATE["kb"],
        top_n=params["top_n"],
        tau=params["tau"],
        rho=params["rho"],
        method=params["method"],
        reranker=get_reranker(params["rerank"]),
    )


def describe(assistant, params):
    """The parameters really used (defaults filled in)."""
    return {
        "method": params["method"], "rerank": params["rerank"],
        "tau": assistant.tau, "rho": assistant.rho, "top_n": assistant.top_n,
    }


# ---------- jobs ----------

def run_question(question, params):
    kb = STATE["kb"]
    with LOCK:
        assistant = build_assistant(params)
        terms = set(keywords(question))
        retrieved = [
            {
                "id": article.id,
                "title": article.title,
                "score": round(float(score), 3),
                "covered": bool(assistant.is_covered(terms, article, score)),
            }
            for article, score in search(kb, question, assistant.top_n, assistant.method)
        ]
        answer = assistant.ask(question)
        used = describe(assistant, params)

    return {
        "question": question,
        "params": used,
        "answered": answer.answered,
        "text": answer.text,
        "note": answer.note,
        "sources": [{"id": a.id, "title": a.title, "updated": a.updated} for a in answer.sources],
        "evidence": answer.evidence,
        "retrieved": retrieved,
    }


def summarize(rows):
    """Same numbers as print_report in evaluate.py, as a dictionary."""
    with_gold = [r for r in rows if "recall" in r]
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    answered_gold = [r for r in with_gold if r["answered"]]
    counts = Counter(r["status"] for r in rows)
    a_scores = [r["top_score"] for r in answerable]
    u_scores = [r["top_score"] for r in unanswerable]

    return {
        "questions": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "recall": {str(k): mean([r["recall"][k] for r in with_gold]) for k in KS},
        "mrr": mean([r["rr"] for r in with_gold]),
        "all_gold": mean([r["all_gold"] for r in with_gold]),
        "status": {
            s: counts[s] for s in
            ("good_answer", "incomplete_answer", "false_refusal", "correct_refusal", "false_answer")
        },
        "accuracy": (counts["good_answer"] + counts["correct_refusal"]) / len(rows) if rows else 0.0,
        "answer_rate": mean([r["answered"] for r in answerable]),
        "refusal_rate": mean([not r["answered"] for r in unanswerable]),
        "cited_gold": mean([r["cited_gold"] for r in answered_gold]),
        "keyword_score": mean([r["keyword_score"] for r in answerable if r["answered"]]),
        "top_score_answerable_min": min(a_scores) if a_scores else None,
        "top_score_unanswerable_max": max(u_scores) if u_scores else None,
    }


def run_evaluation(params):
    if not STATE["questions"]:
        raise FileNotFoundError("eval.jsonl not found: start the server with --eval path/to/eval.jsonl")
    with LOCK:
        assistant = build_assistant(params)
        rows = [evaluate_question(assistant, item) for item in STATE["questions"]]
        used = describe(assistant, params)
    return {"params": used, "summary": summarize(rows), "rows": rows}


# ---------- http ----------

def _json_default(value):
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("a JSON object is expected")
        return payload

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            if not INDEX.exists():
                return self._send(500, {"error": "index.html not found next to app.py"})
            return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/config":
            return self._send(200, STATE["config"])
        if path == "/api/questions":
            return self._send(200, [
                {"id": q["id"], "question": q["question"], "answerable": q["answerable"]}
                for q in STATE["questions"]
            ])
        if path.startswith("/api/article/"):
            article = STATE["kb"].by_id.get(path.rsplit("/", 1)[1])
            if not article:
                return self._send(404, {"error": "Unknown article"})
            return self._send(200, {
                "id": article.id, "title": article.title, "category": article.category,
                "updated": article.updated, "content": article.content,
            })
        return self._send(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in ("/api/ask", "/api/evaluate"):
            return self._send(404, {"error": "Not found"})

        try:
            payload = self._read_json()
            params = parse_params(payload)
            question = str(payload.get("question", "")).strip()
            if path == "/api/ask":
                if not question:
                    raise ValueError("Empty question")
                if len(question) > MAX_QUESTION_CHARS:
                    raise ValueError(f"Question too long (max {MAX_QUESTION_CHARS} characters)")
        except ValueError as error:
            return self._send(400, {"error": str(error)})

        try:
            result = run_question(question, params) if path == "/api/ask" else run_evaluation(params)
            return self._send(200, result)
        except FileNotFoundError as error:
            return self._send(404, {"error": str(error)})
        except Exception as error:  # keep the server alive, show the reason in the page
            return self._send(500, {"error": f"{type(error).__name__}: {error}"})


def find_eval_file(eval_arg, kb_dir):
    """--eval if given, else eval.jsonl next to the kb folder, else in the current folder."""
    if eval_arg:
        return Path(eval_arg)
    for candidate in (Path(kb_dir).parent / "eval.jsonl", Path("eval.jsonl")):
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Velmora assistant - web interface")
    parser.add_argument("--kb", default="kb")
    parser.add_argument("--eval", help="eval.jsonl (default: next to the kb folder)")
    parser.add_argument("--method", default="embeddings", choices=["bm25", "embeddings"],
                        help="method selected when the page opens")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model name")
    parser.add_argument("--rerank", default="cross", choices=RERANKERS,
                        help="reranker selected when the page opens")
    parser.add_argument("--offline", action="store_true", help="BM25 only: no model is loaded")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    methods = ["bm25"]
    kb = None
    if not args.offline:
        print("Building the knowledge base with embeddings (once)...")
        try:
            kb = build_knowledge_base(args.kb, use_embeddings=True, model_name=args.model)
            methods = ["bm25", "embeddings"]
        except Exception as error:
            print(f"Embeddings not available ({type(error).__name__}: {error}). Falling back to BM25 only.")
    if kb is None:
        kb = build_knowledge_base(args.kb, use_embeddings=False)

    start_method = args.method if args.method in methods else "bm25"
    start_rerank = args.rerank if start_method == args.method else "none"

    eval_path = find_eval_file(args.eval, args.kb)
    questions = load_questions(eval_path) if eval_path and eval_path.exists() else []
    if not questions:
        print("No eval.jsonl found: the Evaluation tab is disabled (use --eval path/to/eval.jsonl).")

    STATE["kb"] = kb
    STATE["rerankers"] = {}
    STATE["questions"] = questions
    STATE["config"] = {
        "method": start_method,
        "methods": methods,
        "reranker": start_rerank,
        "rerankers": RERANKERS,
        "defaults": {m: {"tau": a.tau, "rho": a.rho}
                     for m in methods for a in [Assistant(kb, method=m)]},
        "top_n": DEFAULT_TOP_N,
        "articles": len(kb.articles),
        "eval_questions": len(questions),
        "eval_file": str(eval_path) if questions else None,
    }

    if start_rerank != "none":
        print(f"Loading the '{start_rerank}' reranker...")
        with LOCK:
            get_reranker(start_rerank)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Ready: http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()