import { describe, expect, it } from 'vitest'
import { localOutputMaximum, windowRemaining } from './tokens'
import fixtures from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/reply-budget.json'

// One fixture, two implementations: this and the macOS app's NativeReplyBudget
// (ReplyBudgetTests.swift) both have to produce every `expected` below. `exact`
// is the server-counted part of the prompt, absent when all of it is estimated.
describe('native and web model-maximum reply budget', () => {
  for (const row of fixtures) it(row.name, () => {
    expect(windowRemaining(row.context, row.prompt, row.ceiling, row.exact ?? 0)).toBe(row.expected)
    expect(localOutputMaximum(row.context)).toBe(row.context > 0 ? row.context : 4096)
  })
})
