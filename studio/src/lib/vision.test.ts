import { describe, expect, it } from 'vitest'
import { DETAIL_LABEL, DETAIL_ORDER, tokenCap, tokensFor, type VisionBudget } from './vision'

const bonsai: VisionBudget = {
  max_pixels: 16777216, min_pixels: 65536, max_edge: null,
  pixels_per_token: 1024, max_tokens: 16384, min_tokens: 64, auto_max_tokens: 4096,
}

describe('image analysis size', () => {
  it('puts automatic resizing first and offers the original explicitly', () => {
    expect(DETAIL_ORDER).toEqual(['auto', 'high', 'low'])
    expect(DETAIL_LABEL.auto).toBe('Auto-resize')
    expect(DETAIL_LABEL.high).toBe('Original')
  })
  it('prices a large camera photograph using the request cap, not the tower maximum', () => {
    expect(tokenCap(bonsai, 'auto')).toBe(4096)
    expect(tokensFor(bonsai, 6720, 4480, 'auto')).toBeLessThanOrEqual(4096)
    expect(tokensFor(bonsai, 6720, 4480, 'high')).toBeGreaterThan(16000)
    expect(tokensFor(bonsai, 6720, 4480, 'low')).toBe(64)
  })
  it('does not inflate the estimate for small images', () => {
    expect(tokensFor(bonsai, 512, 512, 'auto')).toBe(256)
    expect(tokensFor(bonsai, 512, 512, 'high')).toBe(256)
  })
})
