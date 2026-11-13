// TruthLens AI — Verdict Visual Language
// Shape, label, confidence meter, and color representation.

import type { Stance } from '@/lib/answering'
import type { VerdictLabel } from '@/lib/types'
import { cn } from '@/lib/utils'

export const VERDICT_TEXT: Record<VerdictLabel, { label: string; meaning: string; ink: string; wash: string }> = {
  likely_true: {
    label: 'Likely True',
    meaning: 'Verified evidence directly confirms or strongly substantiates the claim.',
    ink: 'text-emerald-700 dark:text-emerald-300',
    wash: 'bg-emerald-500/15',
  },
  likely_false: {
    label: 'Likely False',
    meaning: 'Verified evidence refutes or contradicts the claim.',
    ink: 'text-rose-700 dark:text-rose-300',
    wash: 'bg-rose-500/15',
  },
  uncertain: {
    label: 'Uncertain',
    meaning: 'Evidence is insufficient, conflicting, or inconclusive.',
    ink: 'text-amber-700 dark:text-amber-300',
    wash: 'bg-amber-500/15',
  },
  supported: {
    label: 'Likely True',
    meaning: 'A source supports the claim, none contradicts it.',
    ink: 'text-emerald-700 dark:text-emerald-300',
    wash: 'bg-emerald-500/15',
  },
  contradicted: {
    label: 'Likely False',
    meaning: 'The sources contradict the claim, none supports it.',
    ink: 'text-rose-700 dark:text-rose-300',
    wash: 'bg-rose-500/15',
  },
  contested: {
    label: 'Uncertain',
    meaning: 'The sources disagree with each other on this claim.',
    ink: 'text-amber-700 dark:text-amber-300',
    wash: 'bg-amber-500/15',
  },
  unverifiable: {
    label: 'Uncertain',
    meaning: 'No reliable source addresses this claim.',
    ink: 'text-amber-700 dark:text-amber-300',
    wash: 'bg-amber-500/15',
  },
  error: {
    label: 'Error',
    meaning: "The verifier's answer was unusable. A retry can fix it.",
    ink: 'text-error-ink',
    wash: 'bg-error-wash',
  },
}

export function VerdictMark({
  verdict,
  size = 18,
  className,
}: {
  verdict: VerdictLabel | 'pending'
  size?: number
  className?: string
}) {
  const common = {
    width: size,
    height: size,
    viewBox: '0 0 20 20',
    className: cn('shrink-0', className),
    'aria-hidden': true,
  }

  switch (verdict) {
    case 'likely_true':
    case 'supported':
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="9" fill="#10b981" />
          <path
            d="M5.8 10.4l2.8 2.8 5.6-6.1"
            fill="none"
            stroke="#ffffff"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      )

    case 'likely_false':
    case 'contradicted':
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="9" fill="#f43f5e" />
          <path
            d="M6.6 6.6l6.8 6.8M13.4 6.6l-6.8 6.8"
            stroke="#ffffff"
            strokeWidth="2.2"
            strokeLinecap="round"
          />
        </svg>
      )

    case 'uncertain':
    case 'contested':
    case 'unverifiable':
      return (
        <svg {...common}>
          <circle cx="10" cy="10" r="9" fill="#f59e0b" />
          <path
            d="M7.9 7.5a2.2 2.2 0 1 1 3.2 2c-.7.4-1.1.8-1.1 1.6"
            fill="none"
            stroke="#ffffff"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle cx="10" cy="14" r="1.1" fill="#ffffff" />
        </svg>
      )

    case 'error':
      return (
        <svg {...common}>
          <rect x="1.5" y="1.5" width="17" height="17" rx="2.5" fill="#ef4444" stroke="#b91c1c" strokeWidth="1.5" />
          <path d="M10 6v5M10 14h.01" stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round" />
        </svg>
      )

    case 'pending':
      return (
        <svg {...common} className={cn('animate-spin opacity-60', className)}>
          <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="30 15" />
        </svg>
      )
  }
}

export function ConfidenceBadge({ confidence }: { confidence?: number }) {
  if (confidence === undefined || confidence === null) return null
  const pct = Math.round(confidence * 100)
  const color =
    pct >= 75
      ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30'
      : pct >= 50
      ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300 border-amber-500/30'
      : 'bg-rose-500/15 text-rose-700 dark:text-rose-300 border-rose-500/30'

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-semibold tracking-wide',
        color,
      )}
      title={`Model confidence: ${pct}%`}
    >
      <span className="size-1.5 rounded-full bg-current opacity-75" />
      {pct}% confidence
    </span>
  )
}

export function AbstentionReasonBadge({ reason }: { reason?: string | null }) {
  if (!reason) return null
  return (
    <div className="mt-2 rounded-md border border-amber-500/20 bg-amber-500/10 px-2.5 py-1 text-xs text-amber-800 dark:text-amber-200">
      <strong className="font-semibold">Uncertainty cause: </strong>
      {reason}
    </div>
  )
}

export function StanceMark({ stance, size = 14 }: { stance: Stance; size?: number }) {
  switch (stance) {
    case 'supports':
      return (
        <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" className="shrink-0">
          <circle cx="8" cy="8" r="7" fill="#10b981" />
          <path d="M4.5 8.2l2.3 2.3 4.7-5" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )
    case 'contradicts':
      return (
        <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" className="shrink-0">
          <circle cx="8" cy="8" r="7" fill="#f43f5e" />
          <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
        </svg>
      )
    case 'silent':
      return (
        <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true" className="shrink-0">
          <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" className="opacity-40" />
        </svg>
      )
    case 'not_read':
      return <span className="text-ink-3">·</span>
  }
}
