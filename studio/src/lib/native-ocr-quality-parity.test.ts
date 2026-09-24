import { describe, expect, it } from 'vitest'
import cases from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/ocr-quality.json'
import { degenerationRatio, DEGENERATION_THRESHOLD } from './ocr'
describe('native/web OCR repetition-review contract', () => {
  for (const row of cases) it(row.name, async () => {
    expect(typeof CompressionStream).toBe('function')
    expect(await degenerationRatio(row.text.repeat(row.repeat)) > DEGENERATION_THRESHOLD).toBe(row.review)
  })
})
