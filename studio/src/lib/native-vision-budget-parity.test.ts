import { describe, expect, it } from 'vitest'
import fixture from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/vision-budget.json'
import { tokensFor, type ImageDetail, type VisionBudget } from './vision'

describe('native and web vision budget contract', () => {
  for (const row of fixture.cases) {
    it(`${row.family} ${row.width}×${row.height} ${row.detail}`, () => {
      const budget = (fixture.budgets as Record<string, VisionBudget>)[row.family]
      expect(tokensFor(budget, row.width, row.height, row.detail as ImageDetail)).toBe(row.expected)
    })
  }
})
