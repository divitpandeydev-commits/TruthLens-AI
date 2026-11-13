// TruthLens AI — Claim Verification Detail Panel
// Displays atomic claim, calibrated verdict, confidence, abstention reason,
// and side-by-side Supporting vs Contradicting evidence panels.

import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { ExternalLink } from 'lucide-react'
import { useState } from 'react'
import { type VerdictEntry, citationMismatch } from '@/lib/answering'
import { shortDoc } from '@/lib/format'
import type { Claim, Passage } from '@/lib/types'
import { cn } from '@/lib/utils'
import { AbstentionReasonBadge, ConfidenceBadge, VERDICT_TEXT, VerdictMark } from '../verdict'

interface Props {
  claim: Claim
  entry?: VerdictEntry
  judging?: boolean
}

export function ClaimDetail({ claim, entry, judging }: Props) {
  const reduce = useReducedMotion()
  const v = entry?.verdict

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.section
        key={claim.id + (v ? ':done' : ':pending')}
        initial={reduce ? false : { opacity: 0, transform: 'translateY(6px)' }}
        animate={{ opacity: 1, transform: 'translateY(0px)' }}
        exit={{ opacity: 0, transition: { duration: 0.08 } }}
        transition={{ duration: 0.22, ease: [0.23, 1, 0.32, 1] }}
        aria-label="Selected claim"
        className="rounded-xl border border-rule bg-paper p-6 shadow-xs"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <VerdictMark verdict={v?.verdict ?? 'pending'} size={28} />
            <div>
              <span className={cn('text-xl font-black tracking-tight', v ? VERDICT_TEXT[v.verdict].ink : 'text-ink-2')}>
                {v ? VERDICT_TEXT[v.verdict].label : judging ? 'Investigating Evidence…' : 'Pending Verification'}
              </span>
              {v && (
                <p className="text-xs text-ink-3">
                  {VERDICT_TEXT[v.verdict].meaning}
                </p>
              )}
            </div>
          </div>
          {v && <ConfidenceBadge confidence={v.confidence} />}
        </div>

        <div className="mt-4 rounded-lg bg-sunk/50 p-3.5">
          <span className="text-xs font-bold uppercase tracking-wider text-ink-3">Atomic Assertion</span>
          <p className="mt-1 text-base font-semibold text-ink leading-snug">{claim.claim}</p>
        </div>

        {v && (
          <>
            <AbstentionReasonBadge reason={v.abstention_reason} />

            <div className="mt-4">
              <span className="text-xs font-bold uppercase tracking-wider text-ink-3">Verifier Reasoning</span>
              <p className="mt-1 text-sm leading-relaxed text-ink-2">{v.justification}</p>
            </div>

            {v.verdict !== 'error' && (
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                <EvidenceSide
                  title="Supporting Evidence"
                  tone="positive"
                  sources={v.supporting_sources}
                  evidence={v.supporting_evidence?.length ? v.supporting_evidence : v.evidence}
                />
                <EvidenceSide
                  title="Contradicting Evidence"
                  tone="negative"
                  sources={v.contradicting_sources}
                  evidence={v.contradicting_evidence?.length ? v.contradicting_evidence : v.evidence}
                />
              </div>
            )}

            {citationMismatch(v) && (
              <p className="mt-4 text-xs text-ink-3">
                Original text cited {shortDoc(citationMismatch(v)!)}, but independent verification consulted external sources.
              </p>
            )}
          </>
        )}
      </motion.section>
    </AnimatePresence>
  )
}

function EvidenceSide({
  title,
  tone,
  sources,
  evidence,
}: {
  title: string
  tone: 'positive' | 'negative'
  sources: string[]
  evidence: Passage[]
}) {
  const [open, setOpen] = useState<string | null>(null)
  const isPos = tone === 'positive'

  // Combine explicit source names with any direct evidence passages
  const effectiveSources = Array.from(new Set([
    ...sources,
    ...((evidence || []).map((p) => p.source_url || p.filename).filter(Boolean) as string[])
  ])).filter((s) => !s.startsWith('[MOCK') && s !== 'refuting-evidence' && s !== 'supporting-evidence')

  return (
    <div className={cn(
      'rounded-xl border p-4',
      isPos ? 'border-emerald-500/20 bg-emerald-500/5' : 'border-rose-500/20 bg-rose-500/5'
    )}>
      <h3 className={cn('text-sm font-bold', isPos ? 'text-emerald-700 dark:text-emerald-300' : 'text-rose-700 dark:text-rose-300')}>
        {isPos ? '✓ ' : '✗ '} {title} ({effectiveSources.length})
      </h3>

      {effectiveSources.length === 0 ? (
        <p className="mt-2 text-xs italic text-ink-3">No {title.toLowerCase()} found in verified web sources.</p>
      ) : (
        <ul className="mt-3 flex flex-col gap-2">
          {effectiveSources.map((s) => {
            const passage = evidence?.find((p) => p.filename === s || p.source_url === s)
            const isUrl = s.startsWith('http://') || s.startsWith('https://') || Boolean(passage?.source_url)
            const targetUrl = s.startsWith('http') ? s : passage?.source_url || s
            const displayTitle = passage?.title || shortDoc(s)

            return (
              <li key={s} className="rounded-lg border border-rule bg-paper p-2.5 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <p className="font-semibold text-ink truncate" title={displayTitle}>
                      {displayTitle}
                    </p>
                    {isUrl && (
                      <p className="text-[11px] text-ink-3 truncate" title={targetUrl}>
                        {targetUrl}
                      </p>
                    )}
                  </div>
                  {isUrl ? (
                    <a
                      href={targetUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 shrink-0 font-medium text-emerald-600 hover:underline"
                    >
                      Visit <ExternalLink className="size-3" />
                    </a>
                  ) : (
                    <button
                      type="button"
                      onClick={() => setOpen(open === s ? null : s)}
                      className="shrink-0 font-medium text-ink-2 hover:text-ink underline"
                    >
                      {open === s ? 'Hide' : 'Quote'}
                    </button>
                  )}
                </div>

                {passage && (
                  <div className="mt-2 border-t border-rule/50 pt-1.5 text-ink-2">
                    <p className="text-[11px] leading-relaxed line-clamp-3">
                      {passage.text}
                    </p>
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
