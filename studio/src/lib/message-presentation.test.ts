import { describe, expect, it } from 'vitest'
import { DEFAULT_PARAMS, type RunMeta, type Usage } from '@/types/chat'
import { answerMetrics, answerMetricsHint, runDetailSections, thinkingLabel, thinkingMetrics, tokenLimitNote } from './message-presentation'

const usage: Usage = { promptTokens: 80, completionTokens: 120, tps: 59.8, ms: 5600, ttftMs: 350,
  answerMs: 2000, reasoningMs: 3000, reasoningTokens: 200, reasoningTps: 66.6, provider: 'Example host', costUsd: 0.012 }
const run: RunMeta = { model: 'model-at-send', spec: 'off', params: { ...DEFAULT_PARAMS, temperature: 0.8, topP: 0.9, topK: 0, maxTokens: 320 }, tools: [], systemPrompt: 'System text', systemPromptName: 'Writing', at: 1,
  gpu: { device: 'Apple M5 Pro', memUsedPeak: 16 * 1024 ** 3, powerPeakW: 40.4, batchPeak: 4, kvPeak: 12, kvTotal: 64, tokSPeak: 119.8 } }

describe('shared message presentation', () => {
  it('shows answer tokens/s but total send-to-done latency and actual cost', () => {
    expect(answerMetrics(usage)).toBe('320 tokens · 57 tok/s · 5.6s · $0.012')
    expect(answerMetricsHint(usage)).toBe('Total time from send to done · 350ms to the first token · served by Example host · 57 tok/s end-to-end')
  })
  it('keeps missing measurements absent and retains cost-only requests', () => {
    expect(answerMetrics()).toBe('')
    expect(answerMetrics({ promptTokens: 5, completionTokens: 0, ms: 3200 })).toBe('')
    expect(answerMetrics({ promptTokens: 5, completionTokens: 0, costUsd: 0.004 })).toBe('$0.004')
    expect(answerMetrics({ promptTokens: 5, completionTokens: 2, answerMs: 125 })).toBe('2 tokens')
    expect(answerMetricsHint()).toBe('')
    expect(runDetailSections({})).toEqual([])
  })
  it('preserves the web transcription footer and thinking labels', () => {
    expect(answerMetrics(usage, 2.35)).toBe('320 tokens · 5.6s (2.4× realtime) · $0.012')
    expect(thinkingLabel(3000)).toBe('Thought for 3.0s')
    expect(thinkingLabel(undefined, true, 2.9)).toBe('Thinking... 2s')
    expect(thinkingMetrics(200, 66.6)).toBe('200 tokens · 67 tok/s')
    expect(thinkingMetrics(200, 66.6, true)).toBe('')
  })
  it('accounts for reasoning in the output limit without inventing a cap', () => {
    expect(tokenLimitNote(usage)).toBe('Reply reached its output limit after 320 generated tokens, including 200 thinking tokens. The context window also includes your prompt and history.')
    expect(tokenLimitNote()).toBe('Reply reached its output limit. The context window also includes your prompt and history.')
  })
  it('uses snapshot provenance and allowlisted metrics with no raw provider data', () => {
    const extra = { ...run, authorization: 'NEVER_PROJECT', response: { api_key: 'NEVER_PROJECT' } }
    const result = runDetailSections({ run: extra, usage })
    expect(result.map(s => s.id)).toEqual(['provenance', 'metrics', 'gpu'])
    expect(result[0].rows).toContainEqual({ label: 'Speculation', value: 'off' })
    expect(result[0].rows).toContainEqual({ label: 'Sampling', value: 'temp 0.8 · top-p 0.9 · top-k off' })
    expect(result[1].rows).toContainEqual({ label: 'Served by', value: 'Example host' })
    expect(result[2].rows).toContainEqual({ label: 'Load', value: '16 GB VRAM · 40 W' })
    expect(JSON.stringify(result)).not.toContain('NEVER_PROJECT')
    expect(JSON.stringify(result)).not.toContain('System text')
  })
})
