"""
Retrieval for TruthLens AI — Query to Passages
==============================================

Supports:
1. Local vector store retrieval (Sentence Transformers / dense search)
2. Hybrid search (meaning + BM25 rank fusion)
3. Cross-encoder reranking
4. Tavily external web search (for live fact-checking of real news claims)
5. Clearly labelled Mock/Offline search (for testing when API keys are not provided)
6. Passage deduplication and relevance ranking
"""

import os
from functools import cached_property
from typing import Protocol, runtime_checkable
import requests

from claimverify.components.embedding import Embedder
from claimverify.components.lexical import Bm25Index, reciprocal_rank_fusion
from claimverify.components.reranking import Reranker
from claimverify.components.store import (
    TOP_K_PER_DOC,
    load_chunk_texts,
    rank_all_per_document,
    search_per_document,
)
from claimverify.contracts import Passage


@runtime_checkable
class Retriever(Protocol):
    """Interface: finds corpus or web passages relevant to a claim/query."""

    def retrieve(self, query: str) -> list[Passage]: ...


def deduplicate_passages(passages: list[Passage], min_length: int = 20) -> list[Passage]:
    """Removes duplicate passages based on text content and source URL/title."""
    seen_texts = set()
    deduped = []
    for p in passages:
        # Normalized snippet fingerprint
        clean_snippet = " ".join(p.text.lower().split()[:25])
        if len(clean_snippet) < min_length:
            continue
        if clean_snippet in seen_texts:
            continue
        seen_texts.add(clean_snippet)
        deduped.append(p)
    return deduped


class LocalStoreRetriever:
    """Retrieves passages from the local store or PostgreSQL using vector similarity."""

    def __init__(self, embedder: Embedder, conn, top_k_per_doc: int = TOP_K_PER_DOC):
        self.embedder = embedder
        self.conn = conn
        self.top_k_per_doc = top_k_per_doc

    def retrieve(self, query: str) -> list[Passage]:
        passages = search_per_document(self.conn, self.embedder.embed_query(query), self.top_k_per_doc)
        return deduplicate_passages(passages)


class PgvectorRetriever(LocalStoreRetriever):
    """Alias for PostgreSQL + pgvector retriever."""
    pass


class HybridRetriever:
    """Implementation: top-k PER document, fusing embedding and BM25 lexical rankings."""

    def __init__(self, embedder: Embedder, conn, top_k_per_doc: int = TOP_K_PER_DOC):
        self.embedder = embedder
        self.conn = conn
        self.top_k_per_doc = top_k_per_doc

    @cached_property
    def _lexical(self) -> tuple[Bm25Index, list[tuple[str, int]]]:
        rows = load_chunk_texts(self.conn)
        return Bm25Index([text for _, _, text in rows]), [(f, i) for f, i, _ in rows]

    def retrieve(self, query: str) -> list[Passage]:
        dense = rank_all_per_document(self.conn, self.embedder.embed_query(query))
        if not dense:
            return []

        bm25, keys = self._lexical
        words = dict(zip(keys, bm25.scores(query), strict=False))

        results = []
        for filename, passages in dense.items():
            by_meaning = [p.chunk_index for p in passages]
            by_words = sorted(by_meaning, key=lambda i: words.get((filename, i), 0.0), reverse=True)
            best = reciprocal_rank_fusion([by_meaning, by_words])[: self.top_k_per_doc]
            chosen = {p.chunk_index: p for p in passages}
            results += [chosen[i] for i in best if i in chosen]

        results.sort(key=lambda p: p.score, reverse=True)
        return deduplicate_passages(results)


class TavilySearchRetriever:
    """Retrieves live web evidence using the free Tavily Search API.
    Falls back gracefully to MockSearchRetriever if TAVILY_API_KEY is not set.
    """

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self.max_results = max_results
        self.fallback = MockSearchRetriever()

    def retrieve(self, query: str) -> list[Passage]:
        if not self.api_key:
            return self.fallback.retrieve(query)

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": self.max_results,
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
        except Exception as e:
            raise RuntimeError(f"Tavily Web Search connection error: {e}") from e

        if resp.status_code != 200:
            raise RuntimeError(f"Tavily Web Search failed (HTTP {resp.status_code}): {resp.text[:200]}")

        data = resp.json()
        results = []
        for idx, r in enumerate(data.get("results", [])):
            score = float(r.get("score", 0.75))
            results.append(
                Passage(
                    filename=r.get("url", f"web-source-{idx}"),
                    source_type="web_search",
                    chunk_index=idx,
                    text=r.get("content", ""),
                    score=round(score, 4),
                    source_url=r.get("url"),
                    title=r.get("title", "Web Evidence"),
                    evidence_id=f"tavily-{idx+1}",
                )
            )
        return deduplicate_passages(results)


class MockSearchRetriever:
    """Offline development retriever with clearly labelled test evidence.
    Ensures the system can be thoroughly developed and evaluated without live API keys.
    """

    MOCK_CORPUS = [
        {
            "title": "NASA James Webb Space Telescope Findings",
            "url": "https://nasa.gov/jwst/exoplanet-atmosphere",
            "text": "The James Webb Space Telescope detected carbon dioxide and water vapor in the atmosphere of exoplanet WASP-96b.",
            "keywords": ["jwst", "telescope", "nasa", "exoplanet", "space", "wasp"],
        },
        {
            "title": "World Health Organization Public Bulletin",
            "url": "https://who.int/news/vaccine-safety-review",
            "text": "Extensive peer-reviewed clinical data confirm mRNA vaccines do not alter human DNA and demonstrate strong safety profiles.",
            "keywords": ["vaccine", "dna", "mrna", "covid", "who", "health", "safety"],
        },
        {
            "title": "MIT Technology Review — Artificial Intelligence",
            "url": "https://technologyreview.com/ai/rag-fact-checking",
            "text": "Retrieval-Augmented Generation can reduce hallucinations in large language models by grounding answers in verified external documents.",
            "keywords": ["rag", "retrieval", "llm", "hallucination", "chunking", "semantic"],
        },
        {
            "title": "Reuters Fact Check Bureau",
            "url": "https://reuters.com/fact-check/solar-energy-grid",
            "text": "Claims that solar panels drain the sun's energy or cause global cooling are scientifically unfounded and false.",
            "keywords": ["solar", "energy", "sun", "cooling", "climate", "grid"],
        },
    ]

    def retrieve(self, query: str) -> list[Passage]:
        q_words = set(query.lower().split())
        scored = []
        for idx, item in enumerate(self.MOCK_CORPUS):
            overlap = len(q_words & set(item["keywords"]))
            text_overlap = len(q_words & set(item["text"].lower().split()))
            score = 0.5 + (0.15 * overlap) + (0.1 * text_overlap)
            score = min(0.98, max(0.40, score))
            scored.append(
                Passage(
                    filename=f"[MOCK / OFFLINE] {item['title']}",
                    source_type="mock_evidence",
                    chunk_index=idx,
                    text=f"[MOCK EVIDENCE / OFFLINE TEST DATA - DO NOT PRESENT AS REAL FACT]: {item['text']}",
                    score=round(score, 4),
                    source_url=item["url"],
                    title=f"[MOCK] {item['title']}",
                    evidence_id=f"mock-{idx+1}",
                )
            )
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:3]


class HybridWebRetriever:
    """Combines local corpus RAG retrieval with Tavily/Mock web search for comprehensive coverage."""

    def __init__(self, local_retriever: Retriever, web_retriever: Retriever):
        self.local_retriever = local_retriever
        self.web_retriever = web_retriever

    def retrieve(self, query: str) -> list[Passage]:
        local_results = self.local_retriever.retrieve(query)
        web_results = self.web_retriever.retrieve(query)
        combined = local_results + web_results
        combined.sort(key=lambda p: p.score, reverse=True)
        return deduplicate_passages(combined)


class RerankingRetriever:
    """Wraps a retriever with cross-encoder reranking."""

    def __init__(self, base: Retriever, reranker: Reranker, top_k_per_doc: int = TOP_K_PER_DOC):
        self.base = base
        self.reranker = reranker
        self.top_k_per_doc = top_k_per_doc

    def retrieve(self, query: str) -> list[Passage]:
        passages = self.base.retrieve(query)
        if not passages:
            return []

        by_doc: dict[str, list[Passage]] = {}
        for p in passages:
            by_doc.setdefault(p.filename, []).append(p)

        results = []
        for group in by_doc.values():
            scores = self.reranker.scores(query, [p.text for p in group])
            order = sorted(range(len(group)), key=lambda j: scores[j], reverse=True)
            results += [group[j] for j in order[: self.top_k_per_doc]]

        results.sort(key=lambda p: p.score, reverse=True)
        return deduplicate_passages(results)


def format_evidence(passages: list[Passage]) -> str:
    """Formats evidence items into traceable text blocks for LLM prompt evaluation."""
    blocks = []
    for idx, p in enumerate(passages, start=1):
        src_label = p.source_url or p.filename
        title_label = f" ({p.title})" if p.title else ""
        blocks.append(
            f"[Evidence #{idx} | Source: {src_label}{title_label} | Score: {p.score:.2f}]\n{p.text}"
        )
    return "\n\n---\n\n".join(blocks)
