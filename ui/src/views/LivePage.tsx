// TruthLens AI — Live News & Claim Verification Page

import { AlertCircle, CheckCircle2, ChevronRight, Cpu, Globe, Play } from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router'
import { startLiveRun, useLiveInfo } from '@/lib/api'
import { cn } from '@/lib/utils'
import { Notice } from '@/components/Notice'

const PRESET_CLAIMS = [
  {
    category: 'Science',
    claim: 'The James Webb Space Telescope detected carbon dioxide and water vapor in exoplanet WASP-96b atmosphere.',
  },
  {
    category: 'Health',
    claim: 'Extensive clinical data confirm that mRNA vaccines do not alter human DNA and demonstrate strong safety.',
  },
  {
    category: 'AI / Tech',
    claim: 'Does semantic chunking improve retrieval performance compared to fixed-size chunking?',
  },
  {
    category: 'Debunking',
    claim: 'New solar panel installations are directly draining energy from the sun and causing global cooling.',
  },
]

export function LivePage() {
  const { data: info, error, isLoading } = useLiveInfo()
  const navigate = useNavigate()
  const [claimText, setClaimText] = useState('')
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)

  const configFile = info?.configs?.[0]?.file || 'config.yaml'

  if (isLoading) return <Notice title="Loading TruthLens AI…" />
  if (error) return <Notice title="API server is not reachable" tone="error" body={(error as Error).message} />
  if (!info?.enabled) {
    return (
      <Notice
        title="Live verification is currently disabled"
        body="The server was started in replay-only mode. Replay existing recorded verification runs."
        action={<Link to="/runs">View Verification History</Link>}
      />
    )
  }

  const config = info.configs?.find((c) => c.file === configFile)
  const isKeyMissing = config?.missing_keys && config.missing_keys.length > 0
  const canStart = claimText.trim().length > 0 && !starting && !info.running

  async function handleVerify() {
    if (!canStart) return
    setStarting(true)
    setStartError(null)
    try {
      const { run_id } = await startLiveRun(claimText.trim(), configFile)
      navigate(`/runs/${run_id}?live=1`)
    } catch (e) {
      setStartError((e as Error).message)
      setStarting(false)
    }
  }

  return (
    <div className="mx-auto grid w-full max-w-[1300px] gap-10 px-4 pt-8 pb-16 sm:px-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <div>
        {!isKeyMissing ? (
          <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3.5 py-1 text-xs font-semibold text-emerald-700 dark:text-emerald-300">
            <span className="size-2 rounded-full bg-emerald-500 animate-pulse" />
            LIVE MODE: Groq ({config?.models?.verify || 'openai/gpt-oss-120b'}) + Tavily Web Search
          </div>
        ) : (
          <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/40 bg-amber-500/10 px-3.5 py-1 text-xs font-semibold text-amber-700 dark:text-amber-300">
            <span className="size-2 rounded-full bg-amber-500" />
            OFFLINE SIMULATION MODE (Keys missing: {config?.missing_keys?.join(', ') || 'none'})
          </div>
        )}

        <h1 className="mt-3 text-3xl font-black tracking-tight text-ink sm:text-4xl">
          Fact-Check News & Claims
        </h1>
        <p className="mt-2 text-base text-ink-2">
          Submit any news article headline, factual statement, or complex claim. TruthLens AI extracts atomic facts, retrieves supporting and contradicting evidence, and issues calibrated verdicts.
        </p>

        <form
          className="mt-6"
          onSubmit={(e) => {
            e.preventDefault()
            handleVerify()
          }}
        >
          <label htmlFor="claim-input" className="block text-sm font-bold text-ink">
            News Article / Claim Text
          </label>
          <textarea
            id="claim-input"
            value={claimText}
            onChange={(e) => setClaimText(e.target.value)}
            rows={5}
            placeholder="Paste a news story, headline, or claim to verify (e.g. 'NASA discovers new signs of life on Mars...')"
            className="mt-2 w-full resize-y rounded-xl border border-rule-strong bg-paper px-4 py-3 text-base text-ink shadow-xs placeholder:text-ink-3 focus:border-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-500/20"
          />

          <div className="mt-2 flex items-center justify-between text-xs text-ink-3">
            <span>{claimText.trim().split(/\s+/).filter(Boolean).length} words</span>
            {claimText && (
              <button
                type="button"
                onClick={() => setClaimText('')}
                className="hover:text-ink underline decoration-dotted"
              >
                Clear input
              </button>
            )}
          </div>

          {/* Quick preset claims */}
          <div className="mt-5">
            <span className="text-xs font-semibold uppercase tracking-wider text-ink-3">
              Sample Verification Presets:
            </span>
            <div className="mt-2 flex flex-col gap-2">
              {PRESET_CLAIMS.map((p) => (
                <button
                  key={p.claim}
                  type="button"
                  onClick={() => setClaimText(p.claim)}
                  className="group flex items-start gap-2.5 rounded-lg border border-rule bg-paper p-2.5 text-left text-xs transition-colors hover:border-emerald-500/40 hover:bg-sunk"
                >
                  <span className="rounded bg-sunk px-1.5 py-0.5 font-bold uppercase text-[10px] tracking-wide text-ink-2 group-hover:bg-emerald-500/10 group-hover:text-emerald-700 dark:group-hover:text-emerald-300">
                    {p.category}
                  </span>
                  <span className="flex-1 text-ink-2 group-hover:text-ink">{p.claim}</span>
                  <ChevronRight className="size-3.5 shrink-0 text-ink-3 opacity-0 group-hover:opacity-100" />
                </button>
              ))}
            </div>
          </div>

          {startError && (
            <div className="mt-4 flex items-center gap-2 rounded-lg border border-rose-500/20 bg-rose-500/10 p-3 text-sm text-rose-800 dark:text-rose-200">
              <AlertCircle className="size-4 shrink-0" />
              <span>{startError}</span>
            </div>
          )}

          <div className="mt-6 flex items-center gap-3">
            <button
              type="submit"
              disabled={!canStart}
              className={cn(
                'inline-flex items-center gap-2 rounded-xl px-6 py-3 text-base font-bold text-white shadow-md transition-all',
                canStart
                  ? 'bg-emerald-600 hover:bg-emerald-500 active:scale-[0.98]'
                  : 'cursor-not-allowed bg-rule-strong text-ink-3 opacity-50',
              )}
            >
              <Play className="size-4.5 fill-current" />
              {starting ? 'Initializing TruthLens AI…' : 'Start Verification'}
            </button>
            {isKeyMissing && (
              <span className="text-xs text-ink-3">
                No API key set in .env — will run in <strong>Offline Simulation Mode</strong>
              </span>
            )}
          </div>
        </form>
      </div>

      {/* Info & Pipeline Architecture Panel */}
      <div className="flex flex-col gap-4">
        <div className="rounded-2xl border border-rule bg-paper p-6 shadow-xs">
          <h2 className="text-lg font-bold text-ink">TruthLens AI Pipeline</h2>
          <p className="mt-1 text-xs text-ink-2">
            Every submission undergoes independent multi-stage verification:
          </p>

          <ol className="mt-4 space-y-3 text-xs">
            <li className="flex items-start gap-2.5">
              <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 font-bold text-emerald-700 dark:text-emerald-300">
                1
              </div>
              <div>
                <strong className="text-ink">Atomic Claim Extraction</strong>
                <p className="text-ink-3">Decomposes complex articles into isolated, testable assertions.</p>
              </div>
            </li>
            <li className="flex items-start gap-2.5">
              <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 font-bold text-emerald-700 dark:text-emerald-300">
                2
              </div>
              <div>
                <strong className="text-ink">Targeted Evidence Retrieval</strong>
                <p className="text-ink-3">Fetches corroborating & refuting sources using hybrid dense & BM25 search.</p>
              </div>
            </li>
            <li className="flex items-start gap-2.5">
              <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 font-bold text-emerald-700 dark:text-emerald-300">
                3
              </div>
              <div>
                <strong className="text-ink">Cross-Evidence Stance & Relevance</strong>
                <p className="text-ink-3">Partitions sources into supporting vs contradicting pools and strips duplicates.</p>
              </div>
            </li>
            <li className="flex items-start gap-2.5">
              <div className="flex size-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 font-bold text-emerald-700 dark:text-emerald-300">
                4
              </div>
              <div>
                <strong className="text-ink">Calibrated Verdict Synthesis</strong>
                <p className="text-ink-3">Issues Likely True, Likely False, or Uncertain with confidence score.</p>
              </div>
            </li>
          </ol>
        </div>

        <div className="rounded-2xl border border-rule bg-paper p-6 shadow-xs">
          <h3 className="text-sm font-bold text-ink">Active System Configuration</h3>
          <div className="mt-3 space-y-2 text-xs text-ink-2">
            <div className="flex items-center justify-between py-1 border-b border-rule">
              <span className="flex items-center gap-1.5"><Cpu className="size-3.5" /> LLM Reasoning</span>
              <span className="font-semibold text-ink">
                {!isKeyMissing ? `Groq (${config?.models?.verify || 'openai/gpt-oss-120b'}) [Live]` : 'Offline Mock'}
              </span>
            </div>
            <div className="flex items-center justify-between py-1 border-b border-rule">
              <span className="flex items-center gap-1.5"><Globe className="size-3.5" /> Web Search API</span>
              <span className="font-semibold text-ink">
                {!isKeyMissing ? 'Tavily Search API [Live]' : 'Offline Local Fallback'}
              </span>
            </div>
            <div className="flex items-center justify-between py-1">
              <span className="flex items-center gap-1.5"><CheckCircle2 className="size-3.5" /> Vector Engine</span>
              <span className="font-semibold text-ink">SentenceTransformers & Cosine Sim</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
