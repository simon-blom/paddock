import { describe, expect, it } from 'vitest'
import rows from '../../../apps/macos/Tests/PaddockConversationCoreTests/Fixtures/context-plans.json'
import type { Conversation } from '@/types/chat'
import { contextTokens, planContext, trimIndex, serverCompactThreshold, compactionTarget } from './tokens'

describe('native/web context planning contract', () => {
  for (const row of rows) it(row.name, () => {
    const conv = {
      id: 'context-fixture', model: 'cloud:ep:remote', leafId: 'm3',
      messages: [6000, 6000, 1600, 1600].map((n, i) => ({ id: `m${i}`, parentId: i ? `m${i-1}` : null,
        role: i % 2 ? 'assistant' : 'user', model: 'cloud:ep:remote', content: [{ type: 'text', text: 'a'.repeat(n) }] })),
      ...(row.saved || row.stale ? { summary: 'Brief', summaryCount: 2, summaryLastId: row.stale ? 'other' : 'm1' } : {}),
    } as Conversation
    const threshold = row.server && row.summarize ? serverCompactThreshold(row.context, row.reply) : 0
    const plan = threshold > 0
      ? { from: row.item ? 2 : trimIndex(conv, row.context, row.reply), summary: undefined }
      : planContext(conv, row.context, row.reply, row.summarize)
    expect(plan.from).toBe(row.from)
    expect(plan.summary).toBe(row.summary)
    expect(threshold).toBe(row.threshold)
    expect(contextTokens(conv)).toBe(row.used)
    expect(compactionTarget(conv, row.context, row.reply)).toBe(row.target)
  })
})
