import { describe, expect, it } from 'vitest'
import type { Message } from '@/types/chat'
import { restoredGraphRuns } from './history'

function message(cypher: unknown, model = 'qwen'): Message {
  return { id: 'answer', role: 'assistant', model, content: [], createdAt: 0, toolCalls: [{
    id: 'call', name: 'graph_query', serverLabel: 'graph', arguments: JSON.stringify({ cypher }), status: 'completed',
  }] } as Message
}
describe('restored graph queries', () => {
  it('keeps the latest twenty queries and model provenance in order', () => {
    const runs = restoredGraphRuns(Array.from({ length: 25 }, (_, i) => message(`RETURN ${i}`)))
    expect(runs).toHaveLength(20)
    expect(runs[0]).toEqual({ cypher: 'RETURN 5', model: 'qwen', response: null })
    expect(runs[runs.length - 1]?.cypher).toBe('RETURN 24')
  })
  it('starts at the most recent attached graph', () => {
    const attachment = { id: 'new', role: 'user', content: [{ type: 'graph', attachmentId: 'new', name: 'new.tvdb' }] } as Message
    expect(restoredGraphRuns([message('RETURN 0'), attachment, message('RETURN 1')])).toEqual([
      { cypher: 'RETURN 1', model: 'qwen', response: null },
    ])
  })
  it('accepts the minimal native projection but rejects malformed/oversized arguments', () => {
    const broken = message('RETURN 0')
    broken.toolCalls![0]!.arguments = '{'
    const unrelated = message('RETURN 2')
    unrelated.toolCalls![0]!.serverLabel = 'other'
    expect(restoredGraphRuns([broken, unrelated, message(null), message(4), message(''), message('🦊'.repeat(20000)), message('RETURN 1')])).toEqual([
      { cypher: 'RETURN 1', model: 'qwen', response: null },
    ])
  })
})
