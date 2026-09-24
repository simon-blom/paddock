import { describe, expect, it } from 'vitest'
import { resolveImageSeed } from './imagine'

describe('web/native image seed policy', () => {
  it('keeps explicit seeds, including zero, for reference edits', () => {
    expect(resolveImageSeed(42, 42, undefined, true, () => 99)).toBe(42)
    expect(resolveImageSeed(0, 42, 'comparison', true, () => 99)).toBe(0)
  })
  it('retains the thread seed for text-only variations, but not edits', () => {
    expect(resolveImageSeed('thread', 42, undefined, false, () => 99)).toBe(42)
    expect(resolveImageSeed('thread', 42, undefined, true, () => 99)).toBe(99)
  })
  it('draws for new threads, retries without a prior seed, and random mode', () => {
    expect(resolveImageSeed('thread', undefined, undefined, false, () => 99)).toBe(99)
    expect(resolveImageSeed('random', 42, undefined, false, () => 99)).toBe(99)
  })
  it('shares compare draws independently of lane completion and reference seeds', () => {
    expect(resolveImageSeed('thread', 42, 'comparison', true, () => 99)).toBe(754347588)
    expect(resolveImageSeed('thread', 12, 'comparison', true, () => 20)).toBe(754347588)
    // UTF-16 hash golden shared with Swift, including a surrogate pair.
    expect(resolveImageSeed('thread', 42, 'compare-🌱', true, () => 99)).toBe(193383420)
  })
})
