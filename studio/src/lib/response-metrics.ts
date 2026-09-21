import type { Usage } from '@/types/chat'

export interface EngineTiming {
  source: string
  version: number
  queue_ms: number
  prefill_ms: number
  decode_ms: number
}

/** No delta-arrival phase inference. Tool arguments can arrive as one chunk;
 * usage counts all rounds, so its denominator must cover the same work. */
export function measuredUsage(raw: Usage): Usage {
  const u = { ...raw }
  delete u.tps
  delete u.answerMs
  delete u.reasoningMs
  delete u.reasoningTps
  const engine = u.timingSource === 'engine'
  u.timingSource = engine ? 'engine' : 'end-to-end'
  const ms = engine ? u.decodeMs : u.ms
  const n = (u.completionTokens ?? 0) + (u.reasoningTokens ?? 0)
  if (ms != null && Number.isFinite(ms) && ms > 0 && Number.isFinite(n) && n > 0) u.tps = n * 1000 / ms
  return u
}

export function engineFields(t?: EngineTiming, previous?: Usage): Partial<Usage> {
  const hasPrevious = (previous?.completionTokens ?? 0) + (previous?.reasoningTokens ?? 0) > 0
  if (t?.source !== 'engine' || t.version !== 1 || !Number.isFinite(t.decode_ms) || t.decode_ms < 0
    || (hasPrevious && previous?.timingSource !== 'engine')) return { timingSource: 'end-to-end' }
  return { timingSource: 'engine', decodeMs: t.decode_ms + (previous?.decodeMs ?? 0),
    prefillMs: t.prefill_ms + (previous?.prefillMs ?? 0), queueMs: t.queue_ms + (previous?.queueMs ?? 0) }
}

export function measuredSpeed(u: Usage): string {
  return u.tps ? `${Math.round(u.tps)} tok/s ${u.timingSource === 'engine' ? 'decode' : 'end-to-end'}` : ''
}
