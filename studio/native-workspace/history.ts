import type { Conversation } from '@/types/chat'
import { object, string } from './protocol'

export interface HistoryQuery { search: string; sort: 'newest' | 'oldest' | 'title'; page: number }
export const HISTORY_PAGE_SIZE = 100
export function historyQuery(value: unknown): HistoryQuery {
  const p = object(value), search = string(p.search, 512)
  if (!['newest', 'oldest', 'title'].includes(String(p.sort)) || !Number.isSafeInteger(p.page) || Number(p.page) < 0) throw new Error('Invalid conversation page')
  return { search, sort: p.sort as HistoryQuery['sort'], page: Number(p.page) }
}
/** Filter before paging. The complete library stays in the shared store;
 * only one bounded page and recent rows cross the WebKit/Swift boundary. */
export function historyPage(conversations: Conversation[], query: HistoryQuery) {
  const term = query.search.trim().toLocaleLowerCase()
  const rows = conversations.filter(c => !term || c.title.toLocaleLowerCase().includes(term)).sort((a, b) => {
    if (!!a.pinned !== !!b.pinned) return a.pinned ? -1 : 1
    const order = query.sort === 'title' ? a.title.localeCompare(b.title, undefined, { numeric: true, sensitivity: 'base' })
      : (a.updatedAt - b.updatedAt) * (query.sort === 'oldest' ? 1 : -1)
    return order || a.id.localeCompare(b.id)
  })
  const page = Math.min(query.page, Math.max(0, Math.ceil(rows.length / HISTORY_PAGE_SIZE) - 1))
  return { rows: rows.slice(page * HISTORY_PAGE_SIZE, (page + 1) * HISTORY_PAGE_SIZE), page, matched: rows.length, total: conversations.length, pageSize: HISTORY_PAGE_SIZE }
}
