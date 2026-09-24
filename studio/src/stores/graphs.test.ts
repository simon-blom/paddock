import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { Message } from '@/types/chat'

const fake = vi.hoisted(() => ({
  sessions: [] as { close: ReturnType<typeof vi.fn> }[],
  bridges: [] as { connected: boolean; closed: boolean; query: (cypher: string, model: string) => Promise<unknown> }[],
  open: undefined as (() => Promise<void>) | undefined,
  schema: undefined as (() => Promise<unknown>) | undefined,
  memory: undefined as (() => Promise<number>) | undefined,
  answer: undefined as (() => Promise<unknown>) | undefined,
  cached: true,
}))
vi.mock('@/lib/graph/session', () => ({ GraphSession: class {
  close = vi.fn()
  constructor() { fake.sessions.push(this) }
  async open() { await fake.open?.() }
  async loadCached() { return { ok: fake.cached, nodes: 2, edges: 1 } }
  async seedTvdb() { return { ok: true, nodes: 2, edges: 1 } }
  async estimatedMemory() { return fake.memory ? await fake.memory() : 1024 }
  async schema() { return fake.schema ? await fake.schema() : { labels: [], labels_detail: [], relationship_types: [], relationship_types_detail: [] } }
} }))
vi.mock('@/lib/graph/bridge', () => ({
  GraphBridge: class {
    connected = false
    closed = false
    constructor(_: string, public query: (cypher: string, model: string) => Promise<unknown>) { fake.bridges.push(this) }
    connect() { this.connected = true }
    close() { this.closed = true }
  },
  async answerModelQuery() { return fake.answer ? await fake.answer() : { body: '{}', response: { rows: [] } } },
}))
import { useGraphsStore } from './graphs'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((a, b) => { resolve = a; reject = b })
  return { promise, resolve, reject }
}
function history(cypher: string): Message[] {
  return [{ id: 'a', role: 'assistant', model: 'qwen', content: [], createdAt: 0, toolCalls: [{
    id: 't', serverLabel: 'graph', name: 'graph_query', arguments: JSON.stringify({ cypher }), status: 'completed',
  }] } as Message]
}
beforeEach(() => {
  setActivePinia(createPinia())
  fake.sessions = []; fake.bridges = []; fake.open = undefined; fake.schema = undefined
  fake.memory = undefined; fake.answer = undefined; fake.cached = true
  vi.unstubAllGlobals()
})
describe('graph session ownership shared by native and web', () => {
  it('restores query chips and switches branch history without rebuilding the database', async () => {
    const graph = useGraphsStore()
    await graph.ensure('chat', 'file', 'Graph', undefined, history('RETURN 1'))
    expect(graph.modelRuns[0]).toEqual({ cypher: 'RETURN 1', model: 'qwen', response: null })
    await graph.ensure('chat', 'file', 'Graph') // Opening its attachment badge has no new history.
    expect(graph.modelRuns[0]?.cypher).toBe('RETURN 1')
    await graph.ensure('chat', 'file', 'Graph', undefined, history('RETURN 2'))
    expect(fake.sessions).toHaveLength(1)
    expect(graph.modelRuns.map(r => r.cypher)).toEqual(['RETURN 2'])
    graph.release()
  })
  it('cannot reconnect a session closed while starting', async () => {
    const gate = deferred<void>()
    fake.open = () => gate.promise
    const graph = useGraphsStore(), pending = graph.ensure('old', 'file', 'Old')
    await vi.waitFor(() => expect(fake.sessions).toHaveLength(1))
    graph.release(); gate.resolve()
    await pending
    expect(graph.status).toBe('idle')
    expect(fake.sessions[0]!.close).toHaveBeenCalled()
    expect(fake.bridges).toHaveLength(0)
  })
  it('ignores a stale schema failure after another conversation opens', async () => {
    const gate = deferred<unknown>()
    fake.schema = () => gate.promise
    const graph = useGraphsStore(), old = graph.ensure('old', 'old-file', 'Old')
    await vi.waitFor(() => expect(graph.phase).toBe('reading the schema'))
    fake.schema = undefined
    await graph.ensure('new', 'new-file', 'New')
    gate.reject(new Error('old worker failed'))
    await old
    expect(graph.status).toBe('ready')
    expect(graph.conversationId).toBe('new')
    expect(graph.error).toBe('')
    expect(fake.bridges).toHaveLength(1)
    graph.release()
  })
  it('does not overwrite current memory with an old measurement', async () => {
    const gate = deferred<number>()
    fake.memory = () => gate.promise
    const graph = useGraphsStore(), old = graph.ensure('old', 'file', 'Old')
    await vi.waitFor(() => expect(graph.counts.nodes).toBe(2))
    fake.memory = undefined
    await graph.ensure('new', 'file2', 'New')
    gate.resolve(999999)
    await old
    expect(graph.memBytes).toBe(1024)
    graph.release()
  })
  it('aborts an attachment fetch on close without an error in the next pane', async () => {
    fake.cached = false
    let signal: AbortSignal | undefined
    vi.stubGlobal('fetch', vi.fn((_: string, init: RequestInit) => new Promise((_, reject) => {
      signal = init.signal as AbortSignal
      signal.addEventListener('abort', () => reject(new DOMException('Closed', 'AbortError')))
    })))
    const graph = useGraphsStore(), pending = graph.ensure('chat', 'file', 'Graph')
    await vi.waitFor(() => expect(signal).toBeDefined())
    graph.release()
    await pending
    expect(signal!.aborted).toBe(true)
    expect(graph.status).toBe('idle')
    expect(graph.error).toBe('')
  })
  it('does not append an old model query result to a different graph', async () => {
    const graph = useGraphsStore()
    await graph.ensure('old', 'file', 'Old')
    const gate = deferred<unknown>()
    fake.answer = () => gate.promise
    const query = fake.bridges[0]!.query('RETURN 1', 'qwen')
    await graph.ensure('new', 'file2', 'New')
    gate.resolve({ body: '{}', response: { rows: [] } })
    await expect(query).rejects.toThrow('Graph closed')
    expect(graph.modelRuns).toEqual([])
    graph.release()
  })
})
