import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import type { SavedPrompt } from '@/lib/api'
const api = vi.hoisted(() => ({ list: vi.fn(), save: vi.fn(), remove: vi.fn() }))
vi.mock('@/lib/api', () => ({ promptsApi: api }))
import { usePromptsStore } from '@/stores/prompts'
import { presetText, promptCommand, promptPage } from '../../native-workspace/prompt-library'

describe('shared prompt library and bounded native adapter', () => {
  beforeEach(() => { setActivePinia(createPinia()); vi.resetAllMocks() })
  it('searches all bodies before paging, without projecting full bodies', () => {
    const rows: SavedPrompt[] = Array.from({ length: 1001 }, (_, i) => ({ id: `p-${i}`, name: `Preset ${i}`, body: `body ${i}`, revision: 'r' }))
    expect(promptPage(rows, { search: 'body 1000', page: 0 }).rows[0].id).toBe('p-1000')
    const all = Array.from({ length: 26 }, (_, page) => promptPage(rows, { page }).rows).flat()
    expect(new Set(all.map(r => r.id)).size).toBe(1001)
    expect(all.every(r => !('body' in r))).toBe(true)
    expect(promptPage(rows.slice(0, 1), { page: 90 }).page).toBe(0)
    expect(() => promptPage(rows, { page: -1 })).toThrow()
  })
  it('enforces UTF-8 rather than character limits before writes', () => {
    expect(() => presetText('🙂'.repeat(40_000), 128 * 1024)).toThrow()
    expect(presetText('é', 2)).toBe('é')
  })
  it('publishes acknowledged saves without a second fallible list request', async () => {
    api.save.mockResolvedValue({ ok: true, prompt: { id: 'p', name: 'N', body: 'B', revision: 'new' } })
    const r = await promptCommand('promptSave', { id: 'p', name: 'N', body: 'B', revision: 'old' })
    expect(r.prompt?.revision).toBe('new')
    expect(api.save.mock.calls[0][0].revision).toBe('old')
    expect(api.list).not.toHaveBeenCalled()
    expect(usePromptsStore().prompts[0].revision).toBe('new')
  })
  it('keeps rows on failed save and delete and forwards the reviewed revision', async () => {
    const s = usePromptsStore(); s.prompts = [{ id: 'p', name: 'N', body: 'B', revision: 'old' }]
    api.save.mockRejectedValue(new Error('Conflict')); api.remove.mockRejectedValue(new Error('Conflict'))
    await expect(promptCommand('promptSave', { id: 'p', name: 'N2', body: 'B', revision: 'reviewed' })).rejects.toThrow('Conflict')
    await expect(promptCommand('promptDelete', { id: 'p', revision: 'reviewed' })).rejects.toThrow('Conflict')
    expect(api.remove).toHaveBeenCalledWith('p', 'reviewed')
    expect(s.prompts[0].name).toBe('N')
  })
  it('a list started before a save cannot overwrite its acknowledged row', async () => {
    let release!: (v: SavedPrompt[]) => void
    api.list.mockReturnValue(new Promise(resolve => { release = resolve }))
    const s = usePromptsStore(), refresh = s.refresh()
    api.save.mockResolvedValue({ ok: true, prompt: { id: 'p', name: 'New', body: 'B', revision: 'new' } })
    await s.save('New', 'B', 'p', '')
    release([]); await refresh
    expect(s.prompts[0].name).toBe('New')
  })
})
