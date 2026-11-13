"""
TruthLens AI — Data Contracts
==============================

The objects passed between stages, defined here once. Each stage knows the
others only through these contracts: a stage's implementation (LLM judge,
local rules, mock verifier, Tavily search, local vector store) can be replaced
without touching the rest, as long as it consumes and produces these objects.

    INDEXING    Document -> Chunk
    ANSWERING   question / news -> Passage -> Draft -> Claim -> Verdict -> ArticleVerification

Every contract is a Pydantic model: it validates the data (useful for LLM
outputs) and serializes to JSON, so each stage can run on its own, from the
file the previous stage produced.
"""

from typing import Literal, get_args

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Verdict Definitions
# ---------------------------------------------------------------------------
# TruthLens AI canonical 3-verdict system:
#   likely_true   evidence directly confirms or strongly supports the claim
#   likely_false  evidence clearly refutes or contradicts the claim
#   uncertain     insufficient, contradictory, ambiguous, or unverifiable evidence
TruthLensVerdict = Literal["likely_true", "likely_false", "uncertain"]
TRUTHLENS_VERDICTS: tuple[str, ...] = get_args(TruthLensVerdict)

# Preserved legacy and alias labels for full backward compatibility
# with recorded runs, evaluation benchmarks, and fixtures.
LegacyJudgeLabel = Literal["supported", "contradicted", "contested", "unverifiable"]
JudgeLabel = LegacyJudgeLabel
JUDGE_LABELS: tuple[str, ...] = get_args(LegacyJudgeLabel)
LEGACY_JUDGE_LABELS: tuple[str, ...] = JUDGE_LABELS

# All allowed verdict labels in API responses and storage
VerdictLabel = Literal[
    "likely_true",
    "likely_false",
    "uncertain",
    "supported",
    "contradicted",
    "contested",
    "unverifiable",
    "Supported",
    "Refuted",
    "Uncertain",
    "Likely True",
    "Likely False",
    "error",
]
VERDICT_LABELS: tuple[str, ...] = get_args(VerdictLabel)

# Canonical normalization map
VERDICT_NORMALIZATION: dict[str, TruthLensVerdict | Literal["error"]] = {
    # Canonical TruthLens
    "likely_true": "likely_true",
    "likely_false": "likely_false",
    "uncertain": "uncertain",
    "Likely True": "likely_true",
    "Likely False": "likely_false",
    "Uncertain": "uncertain",
    # Repo 2 mappings
    "supported": "likely_true",
    "contradicted": "likely_false",
    "contested": "uncertain",
    "unverifiable": "uncertain",
    # Repo 3 mappings
    "Supported": "likely_true",
    "Refuted": "likely_false",
    # System
    "error": "error",
}

# Display labels for UI and human-readable reports
VERDICT_DISPLAY_LABELS: dict[str, str] = {
    "likely_true": "Likely True",
    "likely_false": "Likely False",
    "uncertain": "Uncertain",
    "supported": "Likely True",
    "contradicted": "Likely False",
    "contested": "Uncertain",
    "unverifiable": "Uncertain",
    "Supported": "Likely True",
    "Refuted": "Likely False",
    "Uncertain": "Uncertain",
    "error": "Error",
}

# Visual icons used in summaries and reports
VERDICT_ICONS: dict[str, str] = {
    "likely_true": "✓",
    "likely_false": "✗",
    "uncertain": "?",
    "supported": "✓",
    "contradicted": "✗",
    "contested": "⚠",
    "unverifiable": "?",
    "Supported": "✓",
    "Refuted": "✗",
    "Uncertain": "?",
    "error": "!",
}


def normalize_verdict(label: str) -> TruthLensVerdict | Literal["error"]:
    """Converts any supported verdict label into TruthLens canonical label."""
    clean = str(label).strip()
    return VERDICT_NORMALIZATION.get(clean, "uncertain")


# --- Indexing -----------------------------------------------------------------

class Document(BaseModel):
    """A corpus file, converted to plain text."""
    doc_id: str
    filename: str
    source_type: str
    text: str


class CleanedDocument(BaseModel):
    """The cleaner's output: the repaired document, and what each rule changed."""
    document: Document
    characters_before: int
    changes: dict[str, int] = {}  # rule -> number of repairs


class Chunk(BaseModel):
    """A piece of a document, the unit stored and searched."""
    chunk_id: str
    doc_id: str
    filename: str
    source_type: str
    chunk_index: int
    text: str
    tokens: int | None = None  # size in the embedder's tokens (absent from older runs)
    embedding: list[float] | None = None  # filled in by the embedder
    source_url: str | None = None
    title: str | None = None


# --- Answering & Verification -------------------------------------------------

class Passage(BaseModel):
    """A chunk or external web search snippet returned by the retriever."""
    filename: str
    source_type: str
    chunk_index: int = 0
    text: str
    score: float
    source_url: str | None = None
    title: str | None = None
    evidence_id: str | None = None


class Draft(BaseModel):
    """The draft answer or news summary, and the passages it relies on."""
    question: str
    text: str
    passages: list[Passage] = Field(default_factory=list)


# What a claim is about:
#   topic  the subject of the question/news: verified
#   meta   this answer, its passages or its speaker: kept in the run, not verified
ClaimKind = Literal["topic", "meta"]


class SearchQuery(BaseModel):
    """Search query generated from a claim for external web retrieval."""
    query: str
    claim_id: str
    purpose: str = "general"  # 'supporting', 'refuting', 'general'


class Claim(BaseModel):
    """An atomic assertion extracted from the news article or user claim."""
    id: str
    claim: str
    cited_source: str | None = None
    kind: ClaimKind = "topic"
    search_queries: list[str] = Field(default_factory=list)


def claims_to_verify(claims: list[Claim]) -> list[Claim]:
    """The claims the verifier checks: those about the topic."""
    return [c for c in claims if c.kind == "topic"]


class Verdict(BaseModel):
    """The verifier's judgment on a claim, enhanced with confidence and abstention."""
    claim_id: str
    claim: str
    verdict: VerdictLabel
    justification: str
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    abstention_reason: str | None = None
    supporting_sources: list[str] = Field(default_factory=list)
    contradicting_sources: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    original_cited_source: str | None = None
    verifier: str  # e.g. "gemini:gemini-1.5-flash", "mock_verifier", "llm_judge"
    evidence: list[Passage] = Field(default_factory=list)
    supporting_evidence: list[Passage] = Field(default_factory=list)
    contradicting_evidence: list[Passage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_and_normalize(self) -> "Verdict":
        # Separate supporting and contradicting evidence if not already partitioned
        if self.evidence and not (self.supporting_evidence or self.contradicting_evidence):
            sup_names = set(self.supporting_sources)
            con_names = set(self.contradicting_sources)
            sup_list = []
            con_list = []
            for p in self.evidence:
                if p.filename in sup_names or (p.source_url and p.source_url in sup_names):
                    sup_list.append(p)
                elif p.filename in con_names or (p.source_url and p.source_url in con_names):
                    con_list.append(p)
            self.supporting_evidence = sup_list
            self.contradicting_evidence = con_list

        norm = normalize_verdict(str(self.verdict))
        if norm == "likely_false" and not self.contradicting_sources and not self.abstention_reason:
            # allow fallback if general refutation stated
            self.contradicting_sources = ["refuting-evidence"]
        return self


class ArticleVerification(BaseModel):
    """Overall verification report for a full news article or compound claim."""
    run_id: str
    input_text: str
    overall_verdict: TruthLensVerdict
    overall_confidence: float
    summary: str
    claims: list[Claim]
    verdicts: list[Verdict]
    supporting_evidence_count: int
    contradicting_evidence_count: int
    uncertain_count: int
    is_mock: bool = False
