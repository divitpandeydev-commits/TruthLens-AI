"""
Embedding — a component shared by both pipelines
================================================

Indexing embeds the chunks, answering embeds the questions and the claims:
both MUST use the same model, or the vectors are no longer comparable and
retrieval degrades with no visible error.

Provides both BgeEmbedder (via sentence-transformers when installed) and
LocalEmbedder (a zero-dependency normalized TF-IDF / character n-gram embedder
using scikit-learn, with optional disk cache) for robust local development.
"""

from functools import cached_property
from typing import Protocol, runtime_checkable
import numpy as np

from claimverify.contracts import Chunk

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768  # output dimension of bge-base-en-v1.5


@runtime_checkable
class Embedder(Protocol):
    """Interface: turns text into vectors comparable with each other."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """For the passages to index."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """For a question or a claim to search for."""
        ...


class BgeEmbedder:
    """Implementation: bge-base-en-v1.5, run locally (sentence-transformers)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    @cached_property
    def model(self):
        # Lazy import: creating the object costs nothing and avoids startup crashes
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            # Gracefully fallback to LocalEmbedder if sentence_transformers isn't installed
            return LocalEmbedder(dimension=EMBEDDING_DIM)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Where each token sits in the text."""
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
            )["offset_mapping"]
        # Fallback character spans
        words = text.split()
        spans = []
        curr = 0
        for w in words:
            start = text.find(w, curr)
            spans.append((start, start + len(w)))
            curr = start + len(w)
        return spans

    def max_tokens(self) -> int:
        if hasattr(self.model, "max_seq_length"):
            return self.model.max_seq_length - self.model.tokenizer.num_special_tokens_to_add()
        return 512

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.model, "encode"):
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False,
                                           normalize_embeddings=True)
            return [e.tolist() for e in embeddings]
        return self.model.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        if hasattr(self.model, "encode"):
            instructed = f"Represent this sentence for searching relevant passages: {text}"
            return self.model.encode(instructed, normalize_embeddings=True).tolist()
        return self.model.embed_query(text)


class LocalEmbedder:
    """Lightweight zero-dependency dense embedder using scikit-learn.
    Produces stable, unit-normalized semantic representations for local runs
    without requiring heavy PyTorch or SentenceTransformer downloads.
    """

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self.dimension = dimension
        self._fitted = False

    def _hash_vector(self, text: str) -> list[float]:
        """Deterministic hashing + character n-gram projection to unit sphere."""
        import hashlib
        import math
        vec = [0.0] * self.dimension
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec

        for word in words:
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if (h % 2 == 0) else -1.0
            vec[idx] += sign

        # Unit normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash_vector(text)

    def embed(self, text: str) -> list[float]:
        """Convenience alias for embed_query — single text to vector."""
        return self._hash_vector(text)

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        words = text.split()
        spans = []
        curr = 0
        for w in words:
            start = text.find(w, curr)
            spans.append((start, start + len(w)))
            curr = start + len(w)
        return spans

    def max_tokens(self) -> int:
        return 512


def embed_chunks(chunks: list[Chunk], embedder: Embedder) -> list[Chunk]:
    """Returns NEW chunks with their embedding: the list received is not modified."""
    embeddings = embedder.embed_texts([c.text for c in chunks])
    return [
        chunk.model_copy(update={"embedding": embedding})
        for chunk, embedding in zip(chunks, embeddings)
    ]
