// TruthLens AI — Summary Panel with Verification Metrics

import { type AnsweringState, modelsByRole, totalUsage } from '@/lib/answering'
import { formatSeconds, formatTokens } from '@/lib/format'
import { ANSWERING_STAGES, stageColor, tint } from '@/lib/stages'
import { type Timeline, sumSegments } from '@/lib/timeline'
import type { VerdictLabel } from '@/lib/types'
import { VERDICT_TEXT, VerdictMark } from '../verdict'
import { PanelHeading, Pending } from './common'

export function SummaryPanel({ state, timeline }: { state: AnsweringState; timeline: Timeline }) {
  const verdicts = Object.values(state.verdicts)
  if (!state.claims)
    return (
      <>
        <PanelHeading step="5" title="Verification Summary" />
        <Pending label="Available once claims are verified" />
      </>
    )

  const total = timeline.total
  const waited = sumSegments(timeline.segments, 'wait')
  const usage = totalUsage(state)
  const models = modelsByRole(state.config)

  const primaryLabels: VerdictLabel[] = ['likely_true', 'likely_false', 'uncertain']

  return (
    <>
      <PanelHeading step="5" title="Verification Summary" />
      <div className="grid gap-6 md:grid-cols-3">
        <Block title="TruthLens Verdicts">
          <ul className="space-y-2">
            {primaryLabels.map((l) => {
              const n = verdicts.filter((v) => {
                const raw = v.verdict.verdict
                if (l === 'likely_true') return raw === 'likely_true' || raw === 'supported'
                if (l === 'likely_false') return raw === 'likely_false' || raw === 'contradicted'
                if (l === 'uncertain') return raw === 'uncertain' || raw === 'contested' || raw === 'unverifiable'
                return raw === l
              }).length

              return (
                <li key={l} className={n ? '' : 'opacity-40'}>
                  <span className="flex items-center gap-2">
                    <VerdictMark verdict={l} size={18} />
                    <span className="figures w-6 font-bold">{n}</span>
                    <span className="text-sm font-semibold">{VERDICT_TEXT[l].label}</span>
                  </span>
                </li>
              )
            })}
          </ul>
        </Block>

        <Block title="Verification Time">
          <p className="figures text-2xl font-extrabold">{formatSeconds(total)}</p>
          <div className="mt-2 flex h-2.5 overflow-hidden rounded-full bg-sunk" aria-hidden>
            {ANSWERING_STAGES.flatMap(({ id }) => {
              const work = sumSegments(timeline.segments, 'work', id)
              const wait = sumSegments(timeline.segments, 'wait', id)
              return [
                <span key={`${id}-work`} style={{ width: `${(work / total) * 100}%`, background: stageColor(id) }} />,
                wait > 0 && (
                  <span
                    key={`${id}-wait`}
                    className="hatch"
                    style={{ width: `${(wait / total) * 100}%`, '--hatch': tint(stageColor(id)) } as React.CSSProperties}
                  />
                ),
              ]
            })}
          </div>
          {waited > 0 && <p className="mt-1 text-xs text-ink-3">Includes {formatSeconds(waited)} of rate-limit waits</p>}
        </Block>

        <Block title="Pipeline Architecture">
          <dl className="space-y-1 text-xs">
            <div>
              <dt className="text-ink-3">Judge Model</dt>
              <dd className="font-semibold text-ink truncate">{models.verify || 'Evidence-Aware Judge'}</dd>
            </div>
            <div>
              <dt className="text-ink-3">Token Usage</dt>
              <dd className="figures font-semibold text-ink">
                {formatTokens(usage.tokens_in + usage.tokens_out)} tokens
              </dd>
            </div>
            <div>
              <dt className="text-ink-3">Embedder</dt>
              <dd className="font-semibold text-ink truncate">{state.config?.embedding?.model || 'SentenceTransformers'}</dd>
            </div>
          </dl>
        </Block>
      </div>
    </>
  )
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-rule bg-sheet p-4">
      <h3 className="mb-2 text-xs font-bold uppercase tracking-wider text-ink-3">{title}</h3>
      {children}
    </div>
  )
}
