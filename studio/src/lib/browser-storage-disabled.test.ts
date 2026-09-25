import { afterEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { nextTick } from 'vue'
import { uiPreferences } from './ui-preferences'
import { initializeStudioPersistence } from './browser-storage-migration'
import { useSettingsStore } from '@/stores/settings'
import { useModelsStore } from '@/stores/models'
import { useTelemetryStore } from '@/stores/telemetry'

afterEach(() => vi.unstubAllGlobals())
describe('Studio with browser persistence disabled', () => {
  it('boots and saves model, telemetry and settings using SQLite only', async () => {
    const forbidden = new Proxy({}, { get() { throw Error('Browser persistence is disabled') } })
    for (const key of ['localStorage', 'sessionStorage', 'indexedDB', 'caches']) vi.stubGlobal(key, forbidden)
    vi.stubGlobal('window', { get localStorage() { throw Error('Disabled') }, matchMedia: () => ({ matches: false }) })
    vi.stubGlobal('document', { documentElement: { setAttribute: vi.fn() } })
    const db: Record<string, unknown> = { 'studio.pk_theme': 'light', 'studio.pk_max_tokens_v2': '1' }
    const fetcher = vi.fn(async (_url: unknown, init?: RequestInit) => {
      if (init?.method === 'PUT') Object.assign(db, JSON.parse(String(init.body)))
      return Response.json(db)
    })
    vi.stubGlobal('fetch', fetcher)
    await initializeStudioPersistence()
    setActivePinia(createPinia())
    const settings = useSettingsStore(), models = useModelsStore(), telemetry = useTelemetryStore()
    expect(settings.theme).toBe('light'); expect(telemetry.open).toBe(false)
    settings.maxToolCalls = 25
    models.currentId = 'fixture'
    await nextTick(); await uiPreferences.flush()
    expect(db['studio.pk_max_tool_calls']).toBe('25')
    expect(db['studio.pk_model']).toBe('fixture')
    expect(fetcher.mock.calls.every(([url]) => url === '/api/settings')).toBe(true)
  })
})
