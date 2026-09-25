// The ONLY production access to browser persistence: import known legacy data,
// then remove each source after SQLite acknowledges it. Never write it back.
import { preferenceKeys, type PreferenceAPI, uiPreferences } from './ui-preferences'
import { migrateBrowserReads } from './reads-migration'
import { readHistoryApi } from './api'
import { ref } from 'vue'

const hasOwn = (obj: object, key: string) => Object.prototype.hasOwnProperty.call(obj, key)
export const migrationError = ref('')

export async function migratePreferences(data: Record<string, unknown>, backend: PreferenceAPI, source?: Storage) {
  const legacy = new Map<string, string>()
  if (source) for (const key of [...preferenceKeys, 'pk_reads_panel']) {
    const value = source.getItem(key)
    if (value !== null) legacy.set(key, value)
  }
  const snapshot = data.macos_studio_preferences
  const native = snapshot && typeof snapshot === 'object' ? snapshot as Record<string, unknown> : {}
  const patch: Record<string, unknown> = {}
  for (const key of preferenceKeys) {
    let value = typeof native[key] === 'string' ? native[key] as string : legacy.get(key)
    // Preserve explicit native 8192; only the historical web default migrates.
    if (key === 'pk_max_tokens' && value === '8192' && !native[key] && !legacy.has('pk_max_tokens_v2')) value = 'max'
    if (value !== undefined && !hasOwn(data, `studio.${key}`)) patch[`studio.${key}`] = value
  }
  if (!hasOwn(data, 'studio.pk_max_tokens_v2')) patch['studio.pk_max_tokens_v2'] = '1'
  if (legacy.has('pk_reads_panel') && !hasOwn(data, 'readsPanelOpen')) {
    patch.readsPanelOpen = legacy.get('pk_reads_panel') === '1'
  }
  // INSERT-if-absent is atomic. Another tab's existing SQLite preferences win.
  const saved = Object.keys(patch).length ? await backend.importMissing(patch) : data
  if (source) for (const [key, value] of legacy) {
    const target = key === 'pk_reads_panel' ? 'readsPanelOpen' : `studio.${key}`
    if (hasOwn(saved, target) && source.getItem(key) === value) source.removeItem(key)
  }
  return saved
}

export async function initializeStudioPersistence() {
  let source: Storage | undefined
  // Restricted WebKit and browsers with storage disabled are fully supported.
  try { source = window.localStorage } catch { /* no legacy store accessible */ }
  await uiPreferences.initialize((data, backend) => migratePreferences(data, backend, source))
  await retryReadMigration(source)
  // Compiled WASM is reproducible, not user data. Future loads use the shipped
  // asset plus a worker-lifetime module cache, never CacheStorage.
  try { await globalThis.caches?.delete('lector-wasm-v1') } catch { /* optional legacy cache cleanup */ }
}

export async function retryReadMigration(source?: Storage) {
  try {
    if (!source) { try { source = window.localStorage } catch { /* unavailable */ } }
    if (source) await migrateBrowserReads(source, readHistoryApi)
    migrationError.value = ''
  } catch (e) {
    // Damaged old history must not lock users out of all their SQLite data.
    migrationError.value = e instanceof Error ? e.message : 'Old Reads could not be imported; source data was kept'
  }
}
