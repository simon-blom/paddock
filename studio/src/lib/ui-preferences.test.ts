import { afterEach, describe, expect, it, vi } from 'vitest'
import { createPreferences, type PreferenceAPI } from './ui-preferences'
import { migratePreferences } from './browser-storage-migration'

function backend(initial: Record<string, unknown> = {}) {
  const data = { ...initial }
  const api = {
    load: vi.fn(async () => ({ ...data })),
    importMissing: vi.fn(async (patch: Record<string, unknown>) => {
      for (const [key, value] of Object.entries(patch)) if (!(key in data)) data[key] = value
      return { ...data }
    }),
    save: vi.fn(async (patch: Record<string, unknown>) => { Object.assign(data, patch) }),
  } satisfies PreferenceAPI
  return { data, api, prefs: createPreferences(api) }
}
function legacy(initial: Record<string, string>) {
  const data = new Map(Object.entries(initial))
  return { get length() { return data.size }, key: (n: number) => [...data.keys()][n] ?? null,
    getItem: (k: string) => data.get(k) ?? null, removeItem: (k: string) => { data.delete(k) },
    // Migration must never create a second browser copy.
    setItem: () => { throw Error('browser write forbidden') }, clear: () => { throw Error('blanket removal forbidden') },
  } satisfies Storage
}
afterEach(() => vi.useRealTimers())

describe('SQLite UI preferences', () => {
  it('loads before access, coalesces resizing, writes only dirty keys, survives restart', async () => {
    vi.useFakeTimers()
    const { prefs, api, data } = backend({ 'studio.pk_sidebar_width': '0', unrelated: 'retain' })
    await prefs.initialize()
    expect(prefs.getItem('pk_sidebar_width')).toBe('0')
    for (let n = 0; n < 500; n++) prefs.setItem('pk_sidebar_width', String(n))
    expect(api.save).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(200)
    expect(api.save).toHaveBeenCalledExactlyOnceWith({ 'studio.pk_sidebar_width': '499' })
    expect(prefs.pending.value).toBe(false)
    const reopened = createPreferences(api); await reopened.initialize()
    expect(reopened.getItem('pk_sidebar_width')).toBe('499')
    expect(data.unrelated).toBe('retain')
  })
  it('serializes in-flight writes, including edits back to the acknowledged value', async () => {
    const { prefs, api } = backend()
    await prefs.initialize()
    let resolve!: () => void
    api.save.mockImplementationOnce(() => new Promise<void>(r => { resolve = r }))
    prefs.setItem('pk_theme', 'light')
    const first = prefs.flush()
    prefs.setItem('pk_theme', 'dark')
    expect(api.save).toHaveBeenCalledTimes(1)
    resolve(); await first
    expect(api.save).toHaveBeenLastCalledWith({ 'studio.pk_theme': 'dark' })
    expect(prefs.pending.value).toBe(false)
  })
  it('keeps failed changes for explicit retry and does not silently fall back', async () => {
    const { prefs, api } = backend()
    await prefs.initialize(); prefs.setItem('pk_theme', 'light')
    api.save.mockRejectedValueOnce(Error('disk full'))
    await expect(prefs.flush()).rejects.toThrow('disk full')
    expect(prefs.error.value).toBe('disk full'); expect(prefs.pending.value).toBe(true)
    expect(prefs.getItem('pk_theme')).toBe('light')
    await prefs.flush()
    expect(prefs.error.value).toBe(''); expect(prefs.pending.value).toBe(false)
  })
  it('independent clients cannot overwrite unrelated preference keys', async () => {
    const { prefs, api, data } = backend()
    const other = createPreferences(api)
    await prefs.initialize(); await other.initialize()
    prefs.setItem('pk_model', 'model-a'); other.setItem('pk_theme', 'light')
    await Promise.all([prefs.flush(), other.flush()])
    expect(data).toMatchObject({ 'studio.pk_model': 'model-a', 'studio.pk_theme': 'light' })
  })
  it('rejects arbitrary keys and oversized values', () => {
    const { prefs } = backend()
    expect(() => prefs.setItem('apiKey', 'secret')).toThrow()
    expect(() => prefs.setItem('pk_theme', 'x'.repeat(300_000))).toThrow()
  })
})

describe('legacy preference migration', () => {
  it('imports all known UI and Lector data, leaves unrelated browser data alone', async () => {
    const source = legacy({ pk_theme: 'light', pk_sidebar_width: '0', 'lector.annotationPresets.user': '[{"name":"check"}]', pk_reads_panel: '1', unrelated: 'keep' })
    const { prefs, api, data } = backend()
    await prefs.initialize((d, a) => migratePreferences(d, a, source))
    expect(prefs.getItem('pk_sidebar_width')).toBe('0')
    expect(prefs.getItem('lector.annotationPresets.user')).toContain('check')
    expect(source.length).toBe(1); expect(source.getItem('unrelated')).toBe('keep')
    expect(data.readsPanelOpen).toBe(true)
    await prefs.initialize((d, a) => migratePreferences(d, a, source))
    expect(api.importMissing).toHaveBeenCalledTimes(1)
  })
  it('preserves source data on an unsuccessful import, then retries', async () => {
    const source = legacy({ pk_theme: 'light' })
    const { prefs, api } = backend()
    api.importMissing.mockRejectedValueOnce(Error('offline'))
    await expect(prefs.initialize((d, a) => migratePreferences(d, a, source))).rejects.toThrow('offline')
    expect(source.getItem('pk_theme')).toBe('light')
    await prefs.initialize((d, a) => migratePreferences(d, a, source))
    expect(source.length).toBe(0)
  })
  it('does not overwrite a concurrent SQLite change or resurrect tombstones', async () => {
    const source = legacy({ pk_theme: 'light', pk_model: 'old' })
    const { data, api } = backend({ 'studio.pk_model': null })
    const initial = await api.load()
    data['studio.pk_theme'] = 'dark'
    const result = await migratePreferences(initial, api, source)
    expect(result['studio.pk_theme']).toBe('dark')
    expect(result['studio.pk_model']).toBeNull()
  })
  it('clears historical web reply defaults but preserves explicitly chosen native limits', async () => {
    const a = backend(), b = backend({ macos_studio_preferences: { pk_max_tokens: '8192' } })
    await a.prefs.initialize((d, api) => migratePreferences(d, api, legacy({ pk_max_tokens: '8192' })))
    await b.prefs.initialize((d, api) => migratePreferences(d, api))
    expect(a.prefs.getItem('pk_max_tokens')).toBe('max')
    expect(b.prefs.getItem('pk_max_tokens')).toBe('8192')
  })
})
