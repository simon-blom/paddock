import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { DEFAULT_PARAMS, type Conversation } from '@/types/chat'
import { useChatStore } from '@/stores/chat'
import { titleGenerator } from './chat-title'
import { activeChatControllers } from './chat-activity'

const mocks = vi.hoisted(() => ({
  list: vi.fn(), get: vi.fn(), put: vi.fn(), remove: vi.fn(),
  models: { models: [{ id: 'fixture', status: 'ok' }], canChat: () => true, responsesUrl: () => '/fixture', reasoningLadderFor: () => ({ levels: [] }), reasoningStyleFor: () => 'none' },
  settings: { autoTitle: true },
}))
vi.mock('@/lib/api', () => ({ store: { listConversations: mocks.list, getConversation: mocks.get, putConversation: mocks.put, deleteConversation: mocks.remove }, stubFromSummary: (s: object) => ({ ...s, messages: [] }) }))
vi.mock('@/stores/models', () => ({ useModelsStore: () => mocks.models }))
vi.mock('@/stores/settings', () => ({ useSettingsStore: () => mocks.settings }))
function fixture(id = 'a'): Conversation {
  return { id, title: 'Original', model: 'fixture', systemPrompt: '', params: { ...DEFAULT_PARAMS }, createdAt: 1, updatedAt: 2,
    leafId: 'reply', messages: [
      { id: 'user', parentId: null, role: 'user', createdAt: 1, content: [{ type: 'text', text: 'My question' }] },
      { id: 'reply', parentId: 'user', role: 'assistant', createdAt: 2, content: [{ type: 'text', text: 'My answer' }] },
    ] }
}
beforeEach(() => {
  vi.clearAllMocks()
  setActivePinia(createPinia())
  const data = new Map<string, string>()
  vi.stubGlobal('localStorage', { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => data.set(k, v), removeItem: (k: string) => data.delete(k) })
  vi.stubGlobal('window', { setTimeout, clearTimeout })
  mocks.list.mockResolvedValue([fixture('a'), fixture('b')])
  mocks.get.mockImplementation(async id => fixture(id))
  mocks.put.mockResolvedValue(undefined)
  mocks.remove.mockResolvedValue(undefined)
  mocks.settings.autoTitle = true
})
afterEach(() => { titleGenerator.cancel(); activeChatControllers.clear(); vi.unstubAllGlobals() })
describe('shared durable conversation management', () => {
  it('protects live responses but permits deleting stale crash-time streaming flags', async () => {
    const chat = useChatStore(); await chat.hydrate()
    chat.active!.messages[1].streaming = true
    activeChatControllers.set('a', new Set([new AbortController()]))
    await expect(chat.remove('a')).rejects.toThrow('Stop the response')
    expect(mocks.remove).not.toHaveBeenCalled()
    activeChatControllers.clear()
    await chat.remove('a')
    expect(mocks.remove).toHaveBeenCalledWith('a')
  })
  it('generic composer/model edits keep unsent drafts entirely in memory', async () => {
    const chat = useChatStore(); await chat.hydrate()
    const draft = chat.startDraft('fixture')
    mocks.get.mockClear()
    await chat.edit(draft.id, c => { c.model = 'another-model' })
    expect(chat.active?.model).toBe('another-model')
    expect(chat.activeLoadFailed).toBe(false)
    expect(mocks.get).not.toHaveBeenCalled()
    expect(mocks.put).not.toHaveBeenCalled()
  })
  it('renames an unopened chat without losing messages or changing activity order', async () => {
    const chat = useChatStore(); await chat.hydrate()
    await chat.rename('b', 'A better name')
    expect(chat.conversations.find(c => c.id === 'b')).toMatchObject({ title: 'A better name', titleSource: 'manual', updatedAt: 2 })
    expect(mocks.put.mock.calls[mocks.put.mock.calls.length - 1]?.[0].messages).toHaveLength(2)
  })
  it('failed rename rolls back metadata and reports failure', async () => {
    const chat = useChatStore(); await chat.hydrate()
    mocks.put.mockRejectedValueOnce(new Error('Disk full'))
    await expect(chat.rename('a', 'Lost name')).rejects.toThrow('Disk full')
    expect(chat.active?.title).toBe('Original')
    expect(chat.active?.messages).toHaveLength(2)
    await chat.rename('a', 'Recovered')
    expect(chat.active?.title).toBe('Recovered')
  })
  it('failed hydration cannot report a successful rename or overwrite a stub', async () => {
    const chat = useChatStore(); await chat.hydrate()
    mocks.get.mockRejectedValueOnce(new Error('Offline'))
    await expect(chat.rename('b', 'Wrong')).rejects.toThrow('could not be opened')
    expect(mocks.put).not.toHaveBeenCalled()
  })
  it('deletion failure keeps the row and selection; partial bulk failure is explicit', async () => {
    const chat = useChatStore(); await chat.hydrate()
    mocks.remove.mockRejectedValueOnce(new Error('Offline'))
    await expect(chat.remove('a')).rejects.toThrow('Offline')
    expect(chat.activeId).toBe('a')
    expect(chat.conversations).toHaveLength(2)
    mocks.remove.mockImplementation(async id => { if (id === 'b') throw new Error('Offline') })
    await expect(chat.removeMany(['a', 'b'])).rejects.toThrow('1 conversation(s)')
    expect(chat.conversations.map(c => c.id)).toEqual(['b'])
    expect(chat.activeId).toBe('b')
  })
  it('drains an in-flight save before deleting and refuses late resurrection', async () => {
    const chat = useChatStore(); await chat.hydrate()
    let finish!: () => void
    mocks.put.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve }))
    const save = chat.persistNow(chat.active!, true)
    await vi.waitFor(() => expect(finish).toBeTypeOf('function'))
    const remove = chat.remove('a')
    expect(mocks.remove).not.toHaveBeenCalled()
    finish(); await save; await remove
    expect(mocks.remove).toHaveBeenCalledWith('a')
    await expect(chat.persistNow(fixture('a'), true)).rejects.toThrow('changed')
  })
  it('titles PDF/image-only openings, but respects a manual New chat name', async () => {
    const chat = useChatStore(); await chat.hydrate()
    const c = chat.active!
    c.title = 'New chat'
    c.messages[0].content = [{ type: 'file', attachmentId: 'pdf', name: 'Annual report.pdf', mime: 'application/pdf' }]
    chat.maybeTitle(c)
    expect(c.title).toBe('Annual report.pdf')
    expect(c.titleSource).toBe('fallback')
    await chat.rename('a', 'New chat')
    chat.maybeTitle(c)
    expect(c.title).toBe('New chat')
    expect(c.titleSource).toBe('manual')
  })
  it('manual names and old chats never trigger automatic inference', async () => {
    const chat = useChatStore(); await chat.hydrate()
    const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher)
    await chat.generateTitle('a', true)
    await chat.rename('a', 'Manual name')
    await chat.generateTitle('a', true)
    expect(fetcher).not.toHaveBeenCalled()
  })
  it('manual rename wins a delayed generated label', async () => {
    const chat = useChatStore(); await chat.hydrate()
    let finish!: (r: Response) => void
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(resolve => { finish = resolve })))
    const generate = chat.generateTitle('a')
    await vi.waitFor(() => expect(finish).toBeTypeOf('function'))
    await chat.rename('a', 'My own title')
    finish(Response.json({ status: 'completed', output: [{ type: 'message', content: [{ type: 'output_text', text: 'Too late' }] }] }))
    await generate
    expect(chat.active?.title).toBe('My own title')
    expect(chat.active?.titleSource).toBe('manual')
  })
})
