import type { SamplingParams } from '@/types/chat'

/** Presentation policy shared by Vue and the native Swift composer adapter.
 * No transport, DOM, filesystem or model-name inference lives here. */
export interface ReasoningLadder { levels: string[]; off: boolean; opens: string }
export function reasoningOptions(ladders: ReasoningLadder[]): string[] {
  const levels = [...new Set(ladders.flatMap(l => l.levels))]
  const off = ladders.some(l => l.off)
  return off ? levels.length ? ['off', ...levels] : ['off', 'on'] : levels
}
export function reasoningChoice(options: string[], params: Partial<SamplingParams> | undefined, opens: string): string {
  if (params?.thinking === false && options.includes('off')) return 'off'
  if (params?.reasoningEffort && options.includes(params.reasoningEffort)) return params.reasoningEffort
  return options.includes(opens) ? opens : options.find(o => o !== 'off') ?? ''
}
export function reasoningLabel(value: string): string {
  const labels: Record<string, string> = { off: 'None', on: 'Thinking', minimal: 'Minimal', low: 'Low', medium: 'Medium', high: 'High', xhigh: 'Extra High', max: 'Max' }
  return labels[value] ?? value.charAt(0).toUpperCase() + value.slice(1)
}

export const SAMPLER_DIALS = [
  { key: 'temperature', wire: 'temperature', label: 'Temperature', min: 0, max: 2, step: .05, start: .8 },
  { key: 'topP', wire: 'top_p', label: 'Top-p', min: .01, max: 1, step: .01, start: .95 },
  { key: 'topK', wire: 'top_k', label: 'Top-k', min: 0, max: 200, step: 1, start: 0, off: 0 },
  { key: 'minP', wire: 'min_p', label: 'Min-p', min: 0, max: 1, step: .01, start: 0, off: 0 },
  { key: 'presencePenalty', wire: 'presence_penalty', label: 'Presence penalty', min: -2, max: 2, step: .1, start: 0, off: 0 },
  { key: 'repeatPenalty', wire: 'repeat_penalty', label: 'Repeat penalty', min: 1, max: 2, step: .01, start: 1, off: 1 },
] as const
export type DialKey = typeof SAMPLER_DIALS[number]['key']
export const SAMPLER_KEYS = [...SAMPLER_DIALS.map(d => d.key), 'frequencyPenalty', 'seed'] as const
export function samplerIsSet(params?: Partial<SamplingParams>): boolean {
  return !!params && SAMPLER_KEYS.some(key => params[key] != null)
}
export function samplerDefaults(): Partial<SamplingParams> {
  return Object.fromEntries(SAMPLER_KEYS.map(key => [key, null]))
}
/** UI disclosure for the existing relay restrictions, not a request builder. */
export function samplerCaveats(lanes: { id: string; kind?: string }[], p?: Partial<SamplingParams>) {
  const oai = lanes.some(l => l.kind === 'openai'), claude = lanes.some(l => l.kind === 'anthropic')
  const oaiReasoning = lanes.some(l => l.kind === 'openai' && /^(gpt-5|o[134])/.test(l.id.replace(/^cloud:[^:]+:/, '')))
  const claudeThinking = claude && (p?.thinking ?? true)
  const extensions = !!p && (p.minP != null || p.presencePenalty != null || p.repeatPenalty != null)
  const oaiInert = oai && (extensions || p?.topK != null), claudeInert = claude && extensions
  return { oaiReasoning, claudeThinking, oaiInert, claudeInert,
    notes: [oaiReasoning ? 'gpt-5 and o-series models set their own sampling.' : '',
      claudeThinking ? 'Claude models ignore these while thinking is on.' : '',
      oaiInert ? 'OpenAI models take temperature and top-p from here, not the other settings.' : '',
      claudeInert ? 'Claude models take temperature, top-p and top-k, not the other settings.' : ''].filter(Boolean) }
}
export function samplerLabel(key: DialKey, value: number | null | undefined, advertised?: number): string {
  const dial = SAMPLER_DIALS.find(d => d.key === key)!
  const off = 'off' in dial ? dial.off : undefined
  const number = (v: number) => v === off ? 'off' : key === 'topK' ? String(v) : Number.isInteger(v) ? v.toFixed(1) : String(v)
  if (value != null) return number(value)
  const d = advertised ?? off
  return d == null ? 'Model default' : `Default (${number(d)})`
}

/** Validate an entire patch before changing the conversation; invalid input
 * cannot partially save instructions or reset unrelated sampler defaults. */
export function validateSamplingPatch(value: Record<string, unknown>): void {
  for (const [key, v] of Object.entries(value)) {
    if (key === 'thinking' || key === 'preserveThinking') {
      if (typeof v !== 'boolean') throw new Error('Invalid reasoning switch')
    } else if (key === 'reasoningEffort') {
      if (typeof v !== 'string' || v.length > 32) throw new Error('Invalid reasoning effort')
    } else if (key === 'stop') {
      if (!Array.isArray(v) || v.length > 16 || v.some(s => typeof s !== 'string' || s.length > 1024)) throw new Error('Invalid stop sequences')
    } else {
      const dial = SAMPLER_DIALS.find(d => d.key === key)
      if (!dial && !['seed', 'frequencyPenalty', 'thinkingBudget', 'maxTokens'].includes(key)) throw new Error('Unknown sampling parameter')
      if (v === null) continue
      if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error('Invalid sampling value')
      if (dial && (v < dial.min || v > dial.max)) throw new Error(`${dial.label} must be between ${dial.min} and ${dial.max}`)
      if (['seed', 'topK', 'thinkingBudget', 'maxTokens'].includes(key) && !Number.isSafeInteger(v)) throw new Error(`${key} must be a whole number`)
      if (key === 'maxTokens' && v < 1) throw new Error('Invalid reply limit')
      if (key === 'thinkingBudget' && (v < 0 || v > 1048576)) throw new Error('Invalid thinking budget')
      if (key === 'frequencyPenalty' && (v < -2 || v > 2)) throw new Error('Invalid frequency penalty')
    }
  }
}
