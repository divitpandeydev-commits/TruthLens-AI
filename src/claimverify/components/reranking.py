"""
Reranking — rereading each candidate TOGETHER with the claim
============================================================

The embedder is a bi-encoder: it turns the claim and each chunk into vectors
SEPARATELY, then compares them. It captures the topic, not the answer.

A cross-encoder reads the claim and a chunk TOGETHER, in a single input: its
attention links every word of one to every word of the other, and it judges
whether the chunk answers the claim. Too slow for the whole corpus (one model
pass per pair), it only RE-SORTS a short list of candidates brought by the
search.
"""

from functools import cached_property
from typing import Protocol, runtime_checkable

RERANKER_MODEL = "BAAI/bge-reranker-base"


@runtime_checkable
class Reranker(Protocol):
    """Interface: one relevance score per text, for a query."""

    def scores(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Implementation: a sentence-transformers cross-encoder with lexical fallback."""

    def __init__(self, model_name: str = RERANKER_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    @cached_property
    def model(self):
        # Loaded on first use only, with graceful lexical fallback if torch/model unavailable
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(self.model_name, device=self.device)
        except Exception:
            return None

    def scores(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self.model is not None:
            try:
                return [float(s) for s in self.model.predict([(query, t) for t in texts], show_progress_bar=False)]
            except Exception:
                pass
        # Fast lexical term-overlap fallback
        q_words = set(query.lower().split())
        res = []
        for t in texts:
            t_words = set(t.lower().split())
            overlap = len(q_words & t_words) / max(1, len(q_words))
            res.append(float(overlap))
        return res


def rerank(query: str, candidates: list[int], texts: list[str], reranker: Reranker) -> list[int]:
    """Re-sorts `candidates` (indices into `texts`) from most to least relevant."""
    scores = reranker.scores(query, [texts[i] for i in candidates])
    order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
    return [candidates[j] for j in order]
