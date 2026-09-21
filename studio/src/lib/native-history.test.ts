import { describe, expect, it } from 'vitest'
import { historyPage, historyQuery } from '../../native-workspace/history'
import { DEFAULT_PARAMS, type Conversation } from '@/types/chat'

const rows: Conversation[] = Array.from({ length: 1001 }, (_, i) => ({ id: `chat-${i}`, title: `Chat ${i}`, updatedAt: i, createdAt: i, model: 'fixture', params: { ...DEFAULT_PARAMS }, systemPrompt: '', messages: [], pinned: i === 0 }))
describe('bounded native conversation library', () => {
  it('pages the full library, not just its newest 500 entries', () => {
    const first = historyPage(rows, { search: '', sort: 'newest', page: 0 })
    expect(first.total).toBe(1001)
    expect(first.rows).toHaveLength(100)
    expect(first.rows[0].id).toBe('chat-0')
    const ids = new Set(Array.from({ length: 11 }, (_, page) => historyPage(rows, { search: '', sort: 'newest', page }).rows).flat().map(c => c.id))
    expect(ids.size).toBe(1001)
    expect(historyPage(rows, { search: 'Chat 4', sort: 'oldest', page: 0 }).rows[1].id).toBe('chat-40')
  })
  it('uses deterministic sorting, pins first, and clamps after deletion', () => {
    expect(historyPage(rows, { search: '', sort: 'title', page: 0 }).rows.slice(0, 3).map(c => c.id)).toEqual(['chat-0', 'chat-1', 'chat-2'])
    expect(historyPage(rows.slice(0, 2), { search: '', sort: 'newest', page: 10 }).page).toBe(0)
    expect(historyPage(rows, { search: 'not present', sort: 'newest', page: 1 }).rows).toEqual([])
  })
  it('validates commands before changing query state', () => {
    for (const p of [{ search: '', sort: 'weird', page: 0 }, { search: '', sort: 'title', page: -1 }, { search: '', sort: 'title', page: 0.5 }, { search: 'x'.repeat(513), sort: 'title', page: 0 }]) expect(() => historyQuery(p)).toThrow()
    expect(historyQuery({ search: 'hello', sort: 'oldest', page: 3 }).page).toBe(3)
  })
})
