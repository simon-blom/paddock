import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { store } from '@/lib/api'
import { useChatStore } from './chat'

vi.mock('@/lib/api', () => ({ store: { putConversation: vi.fn() }, stubFromSummary: vi.fn() }))
beforeEach(() => {
  setActivePinia(createPinia())
  vi.stubGlobal('localStorage', { setItem: vi.fn(), getItem: () => null, removeItem: vi.fn() })
  vi.stubGlobal('window', { setTimeout, clearTimeout })
  vi.mocked(store.putConversation).mockReset().mockResolvedValue(new Response())
})
afterEach(() => vi.unstubAllGlobals())

describe('whole-conversation durability', () => {
  it('cannot finish an empty draft commit after the accepted first turn', async () => {
    let release!: () => void
    vi.mocked(store.putConversation).mockImplementationOnce(() => new Promise(resolve => { release = () => resolve(new Response()) }))
    const chat = useChatStore(), c = chat.newConversation('fixture')
    await vi.waitFor(() => expect(store.putConversation).toHaveBeenCalledTimes(1))
    chat.addMessage(c, { id: 'turn', role: 'user', content: [{ type: 'text', text: 'durable' }], createdAt: 1 })
    let accepted = false
    const receipt = chat.persistNow(c, true).then(() => { accepted = true })
    await Promise.resolve()
    expect(store.putConversation).toHaveBeenCalledTimes(1)
    expect(accepted).toBe(false)
    release(); await receipt
    expect(accepted).toBe(true)
    expect(store.putConversation).toHaveBeenCalledTimes(2)
    expect(vi.mocked(store.putConversation).mock.lastCall?.[0].messages).toHaveLength(1)
  })
  it('rejects a strict stale-document receipt instead of pretending it saved', async () => {
    const chat = useChatStore(), c = chat.newConversation('fixture')
    await chat.persistNow(c, true)
    await expect(chat.persistNow({ ...c, messages: [] }, true)).rejects.toThrow('conversation changed')
  })
  it('a failed strict write does not poison later saves', async () => {
    const chat = useChatStore(), c = chat.newConversation('fixture')
    await chat.persistNow(c, true)
    vi.mocked(store.putConversation).mockRejectedValueOnce(new Error('disk unavailable'))
    await expect(chat.persistNow(c, true)).rejects.toThrow('disk unavailable')
    await expect(chat.persistNow(c, true)).resolves.toBeUndefined()
  })
})
