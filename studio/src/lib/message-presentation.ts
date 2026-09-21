import { measuredUsage, measuredSpeed } from './response-metrics'
import type { Message, Usage } from '@/types/chat'
import { fmtCost, fmtDuration, fmtVram } from './format'

/** Shared message chrome for the web and native transcripts. These are
 * presentation-only, allowlisted values, never raw responses or credentials.
 * Missing measurements stay missing; total latency is not decode duration. */
export function answerMetrics(raw?: Usage, realtime?: number | null): string {
  const u = raw ? measuredUsage(raw) : undefined
  if (!u || (!u.completionTokens && !u.reasoningTokens && u.costUsd === undefined && !realtime)) return ''
  const parts: string[] = []
  const outputTokens = (u.completionTokens ?? 0) + (u.reasoningTokens ?? 0)
  if (outputTokens) parts.push(`${outputTokens} tokens`)
  // Keep the compact Web Studio footer; measurement basis belongs in the
  // tooltip and RunDetails, without undoing corrected all-round timing.
  if (!realtime && u.tps) parts.push(`${Math.round(u.tps)} tok/s`)
  const time = u.ms ?? u.answerMs
  if (time) {
    const rate = realtime ? ` (${realtime >= 10 ? Math.round(realtime) : realtime.toFixed(1)}× realtime)` : ''
    parts.push(`${fmtDuration(time)}${rate}`)
  }
  if (u.costUsd !== undefined) parts.push(fmtCost(u.costUsd))
  return parts.filter(Boolean).join(' · ')
}

export function answerMetricsHint(raw?: Usage): string {
  const u = raw ? measuredUsage(raw) : undefined
  if (!u?.ms) return ''
  const parts = ['Total time from send to done']
  if (u.ttftMs) parts.push(`${fmtDuration(u.ttftMs)} to the first token`)
  if (u.reasoningMs) parts.push(`${fmtDuration(u.reasoningMs)} thinking`)
  if (u.answerMs) parts.push(`${fmtDuration(u.answerMs)} writing the answer`)
  if (u.provider) parts.push(`served by ${u.provider}`)
  if (u.tps) parts.push(measuredSpeed(u))
  return parts.join(' · ')
}

export function tokenLimitNote(u?: Usage): string {
  const reasoning = u?.reasoningTokens ?? 0
  if (!reasoning || !u?.completionTokens) return 'Reply reached its output limit. The context window also includes your prompt and history.'
  return `Reply reached its output limit after ${reasoning + u.completionTokens} generated tokens, including ${reasoning} thinking tokens. The context window also includes your prompt and history.`
}

export function thinkingLabel(ms?: number, active = false, elapsedSeconds = 0): string {
  if (active) return elapsedSeconds >= 1 ? `Thinking... ${Math.floor(elapsedSeconds)}s` : 'Thinking...'
  return ms != null ? `Thought for ${fmtDuration(ms)}` : 'Thought for a moment'
}

export function thinkingMetrics(tokens?: number, tps?: number, active = false): string {
  if (active || tokens == null) return ''
  return [`${tokens} tokens`, tps ? `${Math.round(tps)} tok/s` : ''].filter(Boolean).join(' · ')
}

export interface RunDetailSection {
  id: string
  title: string
  rows: { label: string; value: string }[]
}

const dial = (n: number | null | undefined): string => n == null ? 'default' : Number.isInteger(n) ? n.toFixed(1) : String(n)
const join = (parts: (string | undefined | false)[]): string => parts.filter(Boolean).join(' · ') || '-'

/** Snapshot provenance and measurements, not today's runner configuration.
 * Only explicit fields enter the native bridge; extra provider payloads cannot
 * accidentally become Swift state by spreading a run/usage object. */
export function runDetailSections(message: Pick<Message, 'run' | 'usage'>): RunDetailSection[] {
  const { run: r } = message
  const u = message.usage ? measuredUsage(message.usage) : undefined
  const sections: RunDetailSection[] = []
  if (r) {
    const p = r.params
    const sampling = join([
      `temp ${dial(p.temperature)}`, `top-p ${dial(p.topP)}`, `top-k ${p.topK === 0 ? 'off' : (p.topK ?? 'default')}`,
      p.minP ? `min-p ${p.minP}` : '', p.presencePenalty ? `pres ${dial(p.presencePenalty)}` : '',
      p.frequencyPenalty ? `freq ${dial(p.frequencyPenalty)}` : '',
      p.repeatPenalty && p.repeatPenalty !== 1 ? `repeat ${dial(p.repeatPenalty)}` : '',
    ])
    const rows = [{ label: 'Model', value: r.model }]
    if (r.spec) rows.push({ label: 'Speculation', value: r.spec })
    rows.push(
      { label: 'System prompt', value: r.systemPromptName || (r.systemPrompt?.trim() ? 'Custom' : 'None') },
      { label: 'Sampling', value: sampling },
      { label: 'Reasoning', value: p.thinking ? p.reasoningEffort : 'off' },
      { label: 'Max tokens', value: `${p.maxTokens ?? '-'}${p.seed != null ? ` · seed ${p.seed}` : ''}` },
    )
    if (r.tools.length) rows.push({ label: 'Tools', value: r.tools.join(', ') })
    if (r.contended) rows.push({ label: 'Concurrency', value: 'Other compare lanes shared the GPU during this run' })
    sections.push({ id: 'provenance', title: 'Provenance', rows })
  }
  if (u) {
    const rows = [
      { label: 'Tokens', value: join([`${u.promptTokens ?? '-'} in`, `${u.completionTokens == null ? '-' : u.completionTokens + (u.reasoningTokens ?? 0)} out`, u.reasoningTokens ? `${u.reasoningTokens} reasoning` : '']) },
      { label: 'Speed', value: join([measuredSpeed(u), u.ttftMs ? `TTFT ${fmtDuration(u.ttftMs)}` : '', u.ms ? `${fmtDuration(u.ms)} total` : '']) },
    ]
    if (u.reasoningMs || u.answerMs) rows.push({ label: 'Phases', value: join([u.reasoningMs ? `${fmtDuration(u.reasoningMs)} thinking` : '', u.answerMs ? `${fmtDuration(u.answerMs)} writing` : '']) })
    if (u.timingSource === 'engine') rows.push({ label: 'Engine timing', value: join([u.queueMs != null ? `${fmtDuration(u.queueMs)} queued` : '', u.prefillMs != null ? `${fmtDuration(u.prefillMs)} prefill` : '', u.decodeMs != null ? `${fmtDuration(u.decodeMs)} decode (all rounds)` : '']) })
    if (u.provider) rows.push({ label: 'Served by', value: u.provider })
    if (u.costUsd !== undefined) rows.push({ label: 'Cost', value: fmtCost(u.costUsd) })
    sections.push({ id: 'metrics', title: 'Metrics', rows })
  }
  const g = r?.gpu
  if (g) {
    const rows = []
    if (g.device) rows.push({ label: 'Device', value: g.device })
    rows.push({ label: 'Load', value: join([g.utilPeak != null ? `${g.utilPeak}% util` : '', g.memUsedPeak ? `${fmtVram(g.memUsedPeak)} VRAM` : '', g.powerPeakW ? `${Math.round(g.powerPeakW)} W` : '', g.tempPeakC ? `${g.tempPeakC}°C` : '']) })
    if (g.batchPeak || g.kvTotal || g.tokSPeak) rows.push({ label: 'Engine', value: join([g.batchPeak ? `batch ${g.batchPeak}` : '', g.kvTotal ? `KV ${g.kvPeak ?? '-'}/${g.kvTotal}` : '', g.tokSPeak ? `${Math.round(g.tokSPeak)} tok/s peak` : '']) })
    sections.push({ id: 'gpu', title: 'GPU environment (peak)', rows })
  }
  return sections
}
