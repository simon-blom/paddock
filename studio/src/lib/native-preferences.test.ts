import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
vi.mock('@/lib/chat-title', () => ({ titleGenerator: { cancel: vi.fn() } }))
import { useSettingsStore } from '@/stores/settings'
import { restorePreferences } from '../../native-workspace/preferences'
import { preferencePresentation, saveStudioPreferences, validatePreferences } from '../../native-workspace/studio-preferences'
import { uiPreferences } from './ui-preferences'

let database: Record<string, unknown>
function sqliteFetch(_url: unknown, init?: RequestInit) {
  if (init?.method === 'PUT') Object.assign(database, JSON.parse(String(init.body)))
  return Promise.resolve(Response.json(database))
}

describe('native preferences use the shared settings store', () => {
  beforeEach(async () => {
    database = { 'studio.pk_max_tokens_v2': '1' }
    vi.stubGlobal('window', { matchMedia: () => ({ matches: false }), addEventListener: vi.fn() })
    vi.stubGlobal('document', { addEventListener: vi.fn() })
    vi.stubGlobal('fetch', vi.fn(sqliteFetch))
    setActivePinia(createPinia()); await restorePreferences()
  })
  afterEach(async () => {
    vi.stubGlobal('fetch', vi.fn(sqliteFetch))
    await uiPreferences.flush()
    vi.unstubAllGlobals()
  })
  it('validates the whole patch before touching live preferences', async () => {
    const s = useSettingsStore()
    await expect(saveStudioPreferences({ changes: { summarize: false, maxTokens: -1 }, expected: { summarize: true, maxTokens: null } })).rejects.toThrow()
    expect(s.summarize).toBe(true)
    for (const v of [{ maxTokens: 1.5 }, { maxToolCalls: 0 }, { apiKey: 'secret' }, { mapTiles: 'javascript:alert(1)' }, { mapTiles: 'https://user:pass@example.invalid' }]) expect(() => validatePreferences(v)).toThrow()
  })
  it('does not clamp explicit limits to the current model and persists only UI keys', async () => {
    await saveStudioPreferences({ changes: { maxTokens: 32768, maxToolCalls: 50 }, expected: { maxTokens: null, maxToolCalls: null } })
    expect(preferencePresentation()).toMatchObject({ maxTokens: 32768, maxToolCalls: 50, mapHost: 'tiles.openfreemap.org' })
    const calls = vi.mocked(fetch).mock.calls
    const put = calls.find(([, init]) => init?.method === 'PUT')!
    const data = JSON.parse(String(put[1]?.body))
    expect(data['studio.pk_max_tokens']).toBe('32768'); expect(database['studio.pk_max_tokens_v2']).toBe('1')
    expect(JSON.stringify(data)).not.toMatch(/credential|apiKey/)
  })
  it('rolls back runtime without a browser fallback on failed persistence', async () => {
    const s = useSettingsStore()
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 503 })))
    await expect(saveStudioPreferences({ changes: { summarize: false, maxToolCalls: 10 }, expected: { summarize: true, maxToolCalls: null } })).rejects.toThrow('could not be saved')
    expect(s.summarize).toBe(true); expect(s.maxToolCalls).toBe(null)
    expect(uiPreferences.getItem('pk_summarize')).toBe('on')
  })
  it('rejects stale drafts rather than overwriting a changed setting', async () => {
    const s = useSettingsStore(); s.maxToolCalls = 5; await nextTick()
    await expect(saveStudioPreferences({ changes: { maxToolCalls: 50 }, expected: { maxToolCalls: null } })).rejects.toThrow('changed')
    expect(s.maxToolCalls).toBe(5)
  })
  it('restores an explicit 8192 from older native snapshots without rerunning the web migration', async () => {
    database = { macos_studio_preferences: { pk_max_tokens: '8192' } }
    vi.stubGlobal('fetch', vi.fn(sqliteFetch))
    await restorePreferences()
    expect(useSettingsStore().maxTokens).toBe(8192)
  })
  it('accepts raster templates and style URLs without fetching either', () => {
    expect(validatePreferences({ mapTiles: 'http://localhost:8080/{z}/{x}/{y}.png' }).mapTiles).toContain('{z}')
    expect(validatePreferences({ mapTiles: 'https://example.invalid/style.json' }).mapTiles).toContain('style.json')
  })
})
