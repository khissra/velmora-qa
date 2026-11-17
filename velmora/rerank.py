"""Optional reranking of candidate articles. Each reranker returns one score per text.

All models run locally (no API key).
"""


class CrossEncoderReranker:
    """Cross-encoder: reads question + text together (~90 Mo)."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-4-v2"):
    
    
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def scores(self, question, texts):
        return self.model.predict([(question, t) for t in texts]).tolist()


class SemanticReranker:
    """Semantic ranking: small bi-encoder trained for question-passage search (~45 Mo)."""

    def __init__(self, model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def scores(self, question, texts):
        query = self.model.encode(question, normalize_embeddings=True)
        docs = self.model.encode(texts, normalize_embeddings=True)
        return (docs @ query).tolist()


class LLMReranker:
    """Tiny local LLM (~1 Go): score = how much it prefers 'Yes' over 'No'."""

    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        self.yes = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no = self.tokenizer.encode("No", add_special_tokens=False)[0]

    def score(self, question, text):
        import torch
        messages = [{
            "role": "user",
            "content": (
                f"Passage: {text[:1000]}\n\nQuestion: {question}\n\n"
                "Does the passage answer the question? Answer only Yes or No."
            ),
        }]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        return (logits[self.yes] - logits[self.no]).item()
    
    def scores(self, question, texts):
        return [self.score(question, t) for t in texts]


def make_reranker(name):
    """name: 'cross', 'llm', 'semantic' or None."""
    return {
        "cross": CrossEncoderReranker,
        "llm": LLMReranker,
        "semantic": SemanticReranker,
    }[name]() if name else None