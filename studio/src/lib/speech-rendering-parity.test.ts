import { describe, expect, it } from 'vitest'
import { renderWords } from './transcript-diff'
import type { TranscriptMeta } from '@/types/chat'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/speech-rendering.json'

const cases: { name: string; meta: TranscriptMeta; text: string; streaming: boolean; expected: unknown }[] = fixtures

describe('Swift and web transcript rendering contract', () => {
  for (const row of cases) {
    it(row.name, () => {
      // useLiveTurn.apply publishes text with empty metadata; enrichment becomes
      // authoritative on finish. Swift must not hide text behind older words.
      const meta = row.streaming ? {} : row.meta
      const actual = renderWords(meta.segments, meta.words, row.text)
      expect(JSON.parse(JSON.stringify(actual))).toEqual(row.expected)
    })
  }
})
