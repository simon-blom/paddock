import { describe, expect, it } from 'vitest'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/ocr-confidence.json'
import { wordsFromLogprobs } from './ocr'

describe('native and web document confidence', () => {
  for (const [index, fixture] of fixtures.entries()) {
    it(`uses the shared word-folding contract ${index}`, () => {
      const words = wordsFromLogprobs(fixture.entries)
      expect(words).toHaveLength(fixture.expected.length)
      words.forEach((word, i) => {
        expect(word.w).toBe(fixture.expected[i].w)
        expect(word.c).toBeCloseTo(fixture.expected[i].c, 9)
      })
    })
  }
})
