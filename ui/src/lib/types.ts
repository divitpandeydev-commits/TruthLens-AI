// Mirrors src/claimverify/contracts.py and events.py (as the API serializes them).

export const VERDICT_LABELS = [
  'likely_true',
  'likely_false',
  'uncertain',
  'supported',
  'contradicted',
  'contested',
  'unverifiable',
  'error',
] as const
export type VerdictLabel = (typeof VERDICT_LABELS)[number]

export interface Passage {
  filename: string
  source_type: string
  chunk_index: number
  text: string
  score: number
  source_url?: string | null
  title?: string | null
  evidence_id?: string | null
  polarity?: 'supporting' | 'contradicting' | 'neutral' | 'irrelevant'
  information_type?: string
  published_date?: string | null
}

export interface Chunk {
  chunk_id: string
  doc_id: string
  filename: string
  source_type: string
  chunk_index: number
  text: string
  tokens?: number | null
  source_url?: string | null
  title?: string | null
}

export interface Draft {
  question: string
  text: string
  passages: Passage[]
}

export interface Claim {
  id: string
  claim: string
  cited_source: string | null
  kind?: 'topic' | 'meta'
  importance?: 'central' | 'secondary'
  parent_headline?: string | null
  search_queries?: string[]
}

export interface Verdict {
  claim_id: string
  claim: string
  verdict: VerdictLabel
  justification: string
  confidence?: number
  evidence_strength?: number
  abstention_reason?: string | null
  supporting_sources: string[]
  contradicting_sources: string[]
  neutral_sources?: string[]
  irrelevant_sources?: string[]
  source_urls?: string[]
  original_cited_source: string | null
  verifier: string
  evidence: Passage[]
  supporting_evidence?: Passage[]
  contradicting_evidence?: Passage[]
  neutral_evidence?: Passage[]
  irrelevant_evidence?: Passage[]
  conflicting_points?: string[]
}

export interface ArticleVerification {
  run_id: string
  input_text: string
  overall_verdict: VerdictLabel
  overall_confidence: number
  summary: string
  claims: Claim[]
  verdicts: Verdict[]
  supporting_evidence_count: number
  contradicting_evidence_count: number
  uncertain_count: number
}

export interface Usage {
  calls: number
  tokens_in: number
  tokens_out: number
  retries: number
  waited_seconds: number
  seconds: number
}

interface EventBase {
  run_id: string
  pipeline: 'indexing' | 'answering'
  time: string
}

export interface DocumentSummary {
  doc_id: string
  filename: string
  source_type: string
  characters: number
}

export interface CleaningSummary {
  filename: string
  characters_before: number
  characters_after: number
  changes: Record<string, number>
}

export type PipelineEvent = EventBase &
  (
    | { type: 'run_started'; config: Config; inputs: Record<string, unknown> }
    | { type: 'run_finished'; summary: Record<string, unknown>; seconds: number }
    | { type: 'documents_loaded'; documents: DocumentSummary[]; seconds: number }
    | { type: 'documents_cleaned'; documents: CleaningSummary[]; seconds: number }
    | { type: 'chunks_built'; chunks: Chunk[]; seconds: number }
    | { type: 'chunks_embedded'; count: number; dimension: number; seconds: number }
    | { type: 'chunks_stored'; count: number; seconds: number }
    | { type: 'question_started'; question_index: number; question: string }
    | { type: 'passages_retrieved'; question_index: number; passages: Passage[]; seconds: number; usage: Usage }
    | { type: 'draft_written'; question_index: number; draft: Draft; seconds: number; usage: Usage }
    | { type: 'claims_extracted'; question_index: number; claims: Claim[]; seconds: number; usage: Usage }
    | {
        type: 'claim_verified'
        question_index: number
        position: number
        total: number
        verdict: Verdict
        usage: Usage
      }
    | { type: 'question_finished'; question_index: number; verdict_counts: Record<string, number>; seconds: number }
    | {
        type: 'llm_waiting'
        question_index: number
        role: string
        model: string
        seconds: number
        attempt: number
        reason: string
      }
  )

export type EventOf<T extends PipelineEvent['type']> = Extract<PipelineEvent, { type: T }>

export interface RunSummary {
  run_id: string
  pipeline: 'indexing' | 'answering'
  time?: string
  seconds?: number
  documents?: number
  chunks?: number
  questions?: string[]
  verdict_counts?: Record<string, number>
  overall_verdict?: VerdictLabel
  overall_confidence?: number
  overall_summary?: string
  status?: 'running' | 'finished' | 'failed'
  error?: { kind: string; message: string }
  source: 'recorded' | 'fixture'
}

export interface RunDetail {
  summary: RunSummary
  events: PipelineEvent[]
  report?: {
    metadata: Record<string, unknown>
    results: Array<{
      query: string
      overall_verdict?: VerdictLabel
      overall_confidence?: number
      overall_summary?: string
      draft_answer: string
      draft_answer_docs_covered: string[]
      claims: Claim[]
      verdicts: Verdict[]
      verdict_counts: Record<string, number>
      timings: Record<string, number>
    }>
  }
}

export interface Config {
  providers?: Record<string, unknown>
  embedding?: { model: string; type?: string; device?: string }
  indexing?: Record<string, unknown>
  answering?: {
    retriever: { top_k_per_doc: number; rerank?: unknown; type?: string }
    drafter: { model: string; provider?: string }
    decomposer: { model: string; provider?: string }
    verifier: {
      type?: string
      retriever: { top_k_per_doc: number; rerank?: unknown }
      judge: { model: string; provider?: string }
    }
  }
}

export interface LiveInfo {
  enabled: boolean
  database_configured?: boolean
  configs?: Array<{
    file: string
    models?: Record<string, string>
    missing_keys?: string[]
    error?: string
  }>
  estimate?: {
    tokens_per_question: number
    basis_runs: number
    daily_limit_note: string
  }
  running?: string | null
}
