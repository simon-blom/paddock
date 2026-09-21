import { describe, expect, it } from 'vitest'
import { reasoningOptions, reasoningChoice, reasoningLabel, samplerDefaults, samplerIsSet, samplerLabel, samplerCaveats, validateSamplingPatch } from './composer-policy'
import { DEFAULT_PARAMS } from '@/types/chat'

describe('shared web/native composer policy', () => {
  const toggle = { levels: [], off: true, opens: '' }
  const ladder = { levels: ['low', 'medium', 'xhigh'], off: true, opens: 'low' }
  it('keeps switch-only, graded, always-thinking and no-thinking surfaces distinct', () => {
    expect(reasoningOptions([toggle])).toEqual(['off', 'on'])
    expect(reasoningOptions([ladder])).toEqual(['off', 'low', 'medium', 'xhigh'])
    expect(reasoningOptions([{ ...ladder, off: false }])).toEqual(ladder.levels)
    expect(reasoningOptions([{ levels: [], opens: '', off: false }])).toEqual([])
  })
  it('unions compare rungs in advertised order without inventing levels', () => {
    expect(reasoningOptions([ladder, { levels: ['low', 'high'], off: false, opens: 'low' }])).toEqual(['off', 'low', 'medium', 'xhigh', 'high'])
    expect(reasoningLabel('xhigh')).toBe('Extra High')
  })
  it('never selects a primary-model rung missing from the armed compare set', () => {
    expect(reasoningChoice(['off', 'on'], {}, 'low')).toBe('on')
    expect(reasoningChoice(['low', 'high'], { thinking: false }, 'low')).toBe('low')
    expect(reasoningChoice(['off', 'low'], { thinking: false }, 'low')).toBe('off')
    expect(reasoningChoice(['low', 'high'], { reasoningEffort: 'high' }, 'low')).toBe('high')
  })
  it('does not turn untouched sampling controls into explicit request values', () => {
    expect(samplerIsSet(DEFAULT_PARAMS)).toBe(false)
    expect(samplerLabel('temperature', null)).toBe('Model default')
    expect(samplerLabel('temperature', null, .6)).toBe('Default (0.6)')
    expect(samplerLabel('topK', null)).toBe('Default (off)')
    expect(samplerLabel('temperature', 0, .6)).toBe('0.0')
  })
  it('lights and resets every sampling dial, not just temperature/top-p', () => {
    for (const key of ['minP', 'presencePenalty', 'frequencyPenalty', 'repeatPenalty', 'seed']) {
      expect(samplerIsSet({ ...DEFAULT_PARAMS, [key]: 1 })).toBe(true)
    }
    const reset = samplerDefaults()
    expect(Object.values(reset).every(v => v === null)).toBe(true)
    expect(reset).not.toHaveProperty('thinking')
    expect(reset).not.toHaveProperty('maxTokens')
  })
  it('rejects invalid settings before any conversation is changed', () => {
    for (const patch of [{ temperature: -1 }, { topP: 2 }, { topK: 1.5 }, { seed: 1.2 }, { minP: NaN }, { maxTokens: 0 }, { thinking: 'true' }, { secret: 1 }, { stop: [1] }]) {
      expect(() => validateSamplingPatch(patch)).toThrow()
    }
    expect(() => validateSamplingPatch({ ...samplerDefaults(), temperature: 0, topP: .95, thinkingBudget: 0, maxTokens: 32 })).not.toThrow()
  })
  it('discloses provider restrictions without applying them to local or OpenRouter lanes', () => {
    expect(samplerCaveats([{ id: 'gpt-5', kind: 'openrouter' }], {}).notes).toEqual([])
    expect(samplerCaveats([{ id: 'cloud:account:gpt-5', kind: 'openai' }], {}).oaiReasoning).toBe(true)
    expect(samplerCaveats([{ id: 'claude', kind: 'anthropic' }], { thinking: false }).claudeThinking).toBe(false)
    expect(samplerCaveats([{ id: 'claude', kind: 'anthropic' }], { thinking: true, minP: .1 }).notes).toHaveLength(2)
  })
})
