# TruthLens AI 🔍
### AI-Powered Fake News & Claim Verification Using LLM and RAG

> **Built on [claim-verify-rag](https://github.com/Benarbiaa/claim-verify-rag) by Emir Benarbia, with evidence-aware verification techniques inspired by [Evidence-Aware-LLM](https://github.com/qzeng16/Evidence-Aware-LLM) (Apache 2.0).**

---

## What is TruthLens AI?

TruthLens AI verifies news articles and factual claims through a fully automated pipeline:

1. **Atomic Claim Extraction** — Decomposes complex articles into isolated, testable assertions
2. **Targeted Evidence Retrieval** — Fetches corroborating & refuting sources via hybrid dense + BM25 search
3. **Cross-Evidence Stance & Relevance** — Partitions sources into supporting vs. contradicting pools and strips duplicates
4. **Calibrated Verdict Synthesis** — Issues `Likely True`, `Likely False`, or `Uncertain` with confidence score + explanation

---

## Verdicts

| Verdict | Icon | Meaning |
|---------|------|---------|
| **Likely True** | ✅ | Supported by reliable evidence with no significant contradictions |
| **Likely False** | ❌ | Contradicted by reliable evidence |
| **Uncertain** | ❓ | Evidence is insufficient, mixed, or contradictory |

---

## Quick Start (Offline Mode — No API Keys Required)

```powershell
# Clone or navigate to the project
cd "c:\Users\divit pandey\Desktop\fake news detection\claim-verify-rag-main"

# Install Python dependencies
pip install pyyaml fastapi uvicorn python-dotenv openai pydantic numpy

# Set Python path
$env:PYTHONPATH = "src"

# Copy environment template (keys are optional for offline mode)
copy .env.example .env

# Index the sample corpus (uses local vector store)
python -m claimverify.indexing.ingest --corpus_dir data/corpus --config config.yaml --db_url local

# Run a verification (offline mock mode, no API keys needed)
python -m claimverify.answering.pipeline --query "mRNA vaccines do not alter human DNA" --config config.yaml

# Run the test suite
python tests/test_truthlens_core.py
```

---

## Full Stack (Backend + Frontend)

### Start the Python API server
```powershell
cd "c:\Users\divit pandey\Desktop\fake news detection\claim-verify-rag-main"
$env:PYTHONPATH = "src"
python -m claimverify.api.server
```

### Start the React frontend (separate terminal)
```powershell
cd "c:\Users\divit pandey\Desktop\fake news detection\claim-verify-rag-main\ui"
npm install      # first time only
npm run dev      # Vite dev server → http://localhost:5173
```

### Build for production
```powershell
cd "c:\Users\divit pandey\Desktop\fake news detection\claim-verify-rag-main\ui"
npm run build    # outputs to ui/dist/
```

---

## API Keys (Optional — Required for Real Verification)

Copy `.env.example` to `.env` and fill in the keys you have:

```env
# Google Gemini API — free tier at https://aistudio.google.com/
GEMINI_API_KEY=your_key_here

# Groq API (alternative LLM) — free tier at https://console.groq.com/
GROQ_API_KEY=your_key_here

# Tavily Search API — free tier (1000 searches/month) at https://tavily.com/
TAVILY_API_KEY=your_key_here

# DB (use 'local' for the built-in JSON vector store; no PostgreSQL needed)
DB_URL=local
```

> **Without API keys:** TruthLens AI runs in **Offline Simulation Mode** — the LLM is mocked (deterministic placeholder responses) and retrieval uses the local vector store + mock evidence. Results are clearly labelled `[MOCK]` and must NOT be presented as real fact-checks.

---

## Architecture

```
claim-verify-rag-main/
├── src/claimverify/
│   ├── contracts.py          # Data models: Verdict, Claim, Passage, ArticleVerification
│   ├── settings.py           # Pydantic config models (Gemini/Groq/Tavily/local)
│   ├── config.py             # DB connection (local or PostgreSQL)
│   ├── config.yaml           # TruthLens AI configuration
│   ├── llm.py                # LLM client (Gemini / Groq / Mock)
│   ├── factory.py            # Build answering/indexing stages
│   ├── judge_eval.py         # Evaluation: Accuracy, Macro-F1, Confusion Matrix
│   ├── evaluation.py         # E2E evaluation runner
│   ├── components/
│   │   ├── embedding.py      # LocalEmbedder (zero-dep) + BgeEmbedder (sentence-transformers)
│   │   ├── store.py          # LocalStore (JSON file) + PostgreSQL/pgvector
│   │   └── reranking.py      # CrossEncoder + lexical fallback
│   ├── answering/
│   │   ├── pipeline.py       # Full 11-step verification pipeline
│   │   ├── retrieval.py      # LocalStore, Tavily, MockSearch, Hybrid retrievers
│   │   ├── decomposition.py  # Claim extraction & search query generation
│   │   ├── verification.py   # EvidenceAwareVerifier (supporting/contradicting evidence)
│   │   └── drafting.py       # Grounded summary drafter
│   ├── indexing/
│   │   ├── ingest.py         # Corpus ingestion pipeline
│   │   └── loading.py        # Document loaders (MD, TXT, PDF)
│   └── api/
│       ├── app.py            # FastAPI application (TruthLens AI)
│       └── live.py           # Live verification endpoints
├── ui/                       # React + Tailwind frontend
│   └── src/
│       ├── views/
│       │   ├── LivePage.tsx  # Main verification form (TruthLens AI branded)
│       │   └── RunPage.tsx   # Run detail + live streaming
│       └── components/
│           ├── verdict.tsx   # VerdictMark, ConfidenceBadge, StanceMark
│           └── panels/
│               ├── ClaimDetail.tsx    # Supporting + Contradicting evidence panels
│               └── SummaryPanel.tsx  # Overall verdict summary
├── data/
│   ├── corpus/               # Sample documents to index (add your own)
│   └── local_store.json      # Local vector store (auto-created on ingest)
├── tests/
│   └── test_truthlens_core.py  # Core test suite (runs offline)
├── .env.example              # Environment template
├── .env                      # Your local keys (gitignored)
└── README_TRUTHLENS.md       # This file
```

---

## Verification Pipeline (11 Steps)

```
Article/Claim Input
        │
        ▼
1. Broad RAG Retrieval (local corpus)
        │
        ▼
2. Grounded Draft Summary
        │
        ▼
3. Atomic Claim Extraction + Search Query Generation
        │
        ▼
4. Per-claim Evidence Retrieval (Tavily or MockSearch)
        │
        ▼
5. Deduplication (Jaccard similarity threshold)
        │
        ▼
6. Relevance Scoring (cosine similarity)
        │
        ▼
7. Stance Classification (supporting / contradicting / uncertain)
        │
        ▼
8. EvidenceAwareVerifier → LLM reasoning with evidence
        │
        ▼
9. Confidence Scoring + Abstention (if evidence < threshold)
        │
        ▼
10. Per-claim Verdict (likely_true / likely_false / uncertain)
        │
        ▼
11. Overall Verdict Synthesis + Explanation
```

---

## Evaluation

```powershell
# Run evaluation on a gold dataset
$env:PYTHONPATH = "src"
python -m claimverify.evaluation run_judge --config config.yaml --db_url local
```

Metrics computed:
- **Accuracy** — Fraction of correct predictions
- **Macro-Precision / Macro-Recall / Macro-F1** — Averaged over all 3 verdict classes
- **Uncertainty Rate** — Fraction of claims where evidence was insufficient
- **Evidence Relevance** — Average cosine score of retrieved passages
- **Confusion Matrix** — 3×3 (Likely True / Likely False / Uncertain)

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 1.5 Flash (free tier) / Groq |
| Embeddings | SentenceTransformers `all-MiniLM-L6-v2` / LocalEmbedder (hash-based fallback) |
| Vector Store | Local JSON (default) / PostgreSQL + pgvector |
| Web Search | Tavily API (free tier) / Mock |
| Backend | Python 3.11+ / FastAPI / Uvicorn |
| Frontend | React 18 / Vite / TypeScript / Tailwind CSS |
| Reranking | CrossEncoder / Lexical term-overlap fallback |

---

## Attribution & License

- **claim-verify-rag** (Primary codebase): [Emir Benarbia](https://github.com/Benarbiaa/claim-verify-rag) — original architecture preserved
- **Evidence-Aware-LLM** (Feature reference): [qzeng16](https://github.com/qzeng16/Evidence-Aware-LLM) — Apache 2.0 License
  - Evidence stance classification approach
  - Abstention / uncertainty handling
  - Evaluation metric structure
- **TruthLens AI modifications**: Extended pipeline, local vector store, mock mode, frontend rebranding

> ⚠️ **Mock Mode Warning:** When running without API keys, all results are simulated placeholders labelled `[MOCK / OFFLINE TEST DATA]`. Do NOT present mock results as real news verification.
