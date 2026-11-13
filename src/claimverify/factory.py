"""
TruthLens AI — Factory
======================

Builds the pipeline's stages from config.yaml.
Supports both live API-backed components (Gemini, Groq, Tavily, PostgreSQL) and
lightweight, zero-cost offline/mock components for instant local development.
"""

import os
from dataclasses import dataclass, field
from functools import cache

from claimverify.answering.decomposition import Decomposer, LLMDecomposer
from claimverify.answering.drafting import Drafter, LLMDrafter
from claimverify.answering.retrieval import (
    HybridRetriever,
    HybridWebRetriever,
    LocalStoreRetriever,
    PgvectorRetriever,
    RerankingRetriever,
    Retriever,
    TavilySearchRetriever,
    WebSearchRetriever,
)
from claimverify.answering.verification import (
    EvidenceAwareVerifier,
    Judge,
    LangGraphVerifier,
    LLMJudge,
    Verifier,
)
from claimverify.components.embedding import BgeEmbedder, Embedder, LocalEmbedder
from claimverify.indexing.chunking import Chunker, FixedSizeChunker, Tokenizer
from claimverify.indexing.cleaning import Cleaner, MinimalCleaner, NoCleaner
from claimverify.indexing.loading import FileLoader, Loader
from claimverify.llm import LLM, RetryPolicy, UsageMeter, make_llm
from claimverify.settings import LLMStageSettings, RetrieverSettings, Settings

# ---------------------------------------------------------------------------
# Registries: type (in config.yaml) -> constructor
# ---------------------------------------------------------------------------

LOADERS = {
    "files": lambda cfg: FileLoader(),
}
CLEANERS = {
    "minimal": lambda cfg: MinimalCleaner(),
    "none": lambda cfg: NoCleaner(),
}
CHUNKERS = {
    "fixed_size": lambda cfg, tokenizer: FixedSizeChunker(tokenizer, cfg.chunk_size, cfg.overlap_ratio),
}
EMBEDDERS = {
    "bge": lambda cfg: BgeEmbedder(cfg.model, cfg.device),
    "sentence_transformers": lambda cfg: BgeEmbedder(cfg.model, cfg.device),
    "local": lambda cfg: LocalEmbedder(),
    "tfidf": lambda cfg: LocalEmbedder(),
    "mock": lambda cfg: LocalEmbedder(),
}
RETRIEVERS = {
    "pgvector": lambda cfg, embedder, conn: PgvectorRetriever(embedder, conn, cfg.top_k_per_doc),
    "hybrid": lambda cfg, embedder, conn: HybridRetriever(embedder, conn, cfg.top_k_per_doc),
    "local": lambda cfg, embedder, conn: LocalStoreRetriever(embedder, conn, cfg.top_k_per_doc),
    "tavily": lambda cfg, embedder, conn: TavilySearchRetriever(max_results=cfg.top_k_per_doc * 3),
    "web_search": lambda cfg, embedder, conn: WebSearchRetriever(
        TavilySearchRetriever(max_results=cfg.top_k_per_doc * 3),
        max_results=cfg.top_k_per_doc * 3,
    ),
    "hybrid_web": lambda cfg, embedder, conn: HybridWebRetriever(
        LocalStoreRetriever(embedder, conn, cfg.top_k_per_doc),
        TavilySearchRetriever(max_results=cfg.top_k_per_doc),
    ),
}
DRAFTERS = {
    "llm": lambda llm: LLMDrafter(llm),
    "mock": lambda llm: LLMDrafter(llm),
}
DECOMPOSERS = {
    "llm": lambda llm: LLMDecomposer(llm),
    "mock": lambda llm: LLMDecomposer(llm),
}
JUDGES = {
    "llm": lambda llm, min_rel=0.35: LLMJudge(llm, min_relevance_score=min_rel),
    "mock": lambda llm, min_rel=0.35: LLMJudge(llm, min_relevance_score=min_rel),
}
VERIFIERS = {
    "evidence_aware": lambda retriever, judge: EvidenceAwareVerifier(retriever, judge),
    "langgraph": lambda retriever, judge: LangGraphVerifier(retriever, judge),
    "mock": lambda retriever, judge: EvidenceAwareVerifier(retriever, judge),
}


def _pick(registry: dict, type_: str, stage: str):
    if type_ not in registry:
        raise ValueError(f"{stage}: unknown type '{type_}' (available: {sorted(registry)})")
    return registry[type_]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_llm(settings: Settings, stage: LLMStageSettings, role: str,
              meter: UsageMeter | None = None) -> LLM:
    if stage.type == "mock":
        return make_llm(role, "mock-model", "mock-url", "mock-key", meter=meter, is_mock=True)

    provider = settings.providers[stage.provider]
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        # Fall back to clearly labelled mock mode so the application can run without API keys
        return make_llm(
            role,
            f"mock-{stage.model}",
            provider.base_url,
            "mock-key",
            meter=meter,
            is_mock=True,
        )

    retry = RetryPolicy(provider.max_retries, provider.max_wait_seconds, provider.timeout_seconds)
    return make_llm(role, stage.model, provider.base_url, api_key, retry, meter, is_mock=False)


def build_embedder(settings: Settings) -> Embedder:
    return _pick(EMBEDDERS, settings.embedding.type, "embedding")(settings.embedding)


def build_loader(settings: Settings) -> Loader:
    cfg = settings.indexing.loader
    return _pick(LOADERS, cfg.type, "indexing.loader")(cfg)


def build_cleaner(settings: Settings) -> Cleaner:
    cfg = settings.indexing.cleaner
    return _pick(CLEANERS, cfg.type, "indexing.cleaner")(cfg)


def build_chunker(settings: Settings, tokenizer: Tokenizer) -> Chunker:
    cfg = settings.indexing.chunker
    return _pick(CHUNKERS, cfg.type, "indexing.chunker")(cfg, tokenizer)


def build_retriever(cfg: RetrieverSettings, embedder: Embedder, conn, stage: str) -> Retriever:
    if cfg.rerank is None:
        return _pick(RETRIEVERS, cfg.type, stage)(cfg, embedder, conn)

    base = _pick(RETRIEVERS, cfg.type, stage)(
        cfg.model_copy(update={"top_k_per_doc": cfg.rerank.candidates}), embedder, conn
    )
    return RerankingRetriever(base, _reranker(cfg.rerank.model, cfg.rerank.device), cfg.top_k_per_doc)


@cache
def _reranker(model: str, device: str):
    from claimverify.components.reranking import CrossEncoderReranker
    return CrossEncoderReranker(model, device)


def build_drafter(settings: Settings, meter: UsageMeter | None = None) -> Drafter:
    cfg = settings.answering.drafter
    return _pick(DRAFTERS, cfg.type, "answering.drafter")(
        build_llm(settings, cfg, "draft", meter)
    )


def build_decomposer(settings: Settings, meter: UsageMeter | None = None) -> Decomposer:
    cfg = settings.answering.decomposer
    return _pick(DECOMPOSERS, cfg.type, "answering.decomposer")(
        build_llm(settings, cfg, "decompose", meter)
    )


def build_judge(settings: Settings, meter: UsageMeter | None = None) -> Judge:
    cfg = settings.answering.verifier.judge
    min_rel = getattr(settings.answering.verifier, "min_relevance_score", 0.35)
    return _pick(JUDGES, cfg.type, "answering.verifier.judge")(
        build_llm(settings, cfg, "verify", meter), min_rel=min_rel
    )


def build_verifier(settings: Settings, embedder: Embedder, conn,
                   meter: UsageMeter | None = None) -> Verifier:
    cfg = settings.answering.verifier
    retriever = build_retriever(cfg.retriever, embedder, conn, "answering.verifier.retriever")
    return _pick(VERIFIERS, cfg.type, "answering.verifier")(retriever, build_judge(settings, meter))


# ---------------------------------------------------------------------------
# Stages Assembler
# ---------------------------------------------------------------------------

@dataclass
class IndexingStages:
    loader: Loader
    cleaner: Cleaner
    chunker: Chunker
    embedder: Embedder


@dataclass
class AnsweringStages:
    retriever: Retriever
    drafter: Drafter
    decomposer: Decomposer
    verifier: Verifier
    meter: UsageMeter = field(default_factory=UsageMeter)


def build_indexing(settings: Settings, embedder: Embedder | None = None) -> IndexingStages:
    embedder = embedder or build_embedder(settings)
    if not isinstance(embedder, Tokenizer):
        raise TypeError(f"{type(embedder).__name__} does not provide token_spans.")
    return IndexingStages(
        loader=build_loader(settings),
        cleaner=build_cleaner(settings),
        chunker=build_chunker(settings, tokenizer=embedder),
        embedder=embedder,
    )


def build_answering(settings: Settings, conn, embedder: Embedder | None = None) -> AnsweringStages:
    embedder = embedder or build_embedder(settings)
    meter = UsageMeter()
    return AnsweringStages(
        retriever=build_retriever(settings.answering.retriever, embedder, conn,
                                  "answering.retriever"),
        drafter=build_drafter(settings, meter),
        decomposer=build_decomposer(settings, meter),
        verifier=build_verifier(settings, embedder, conn, meter),
        meter=meter,
    )
