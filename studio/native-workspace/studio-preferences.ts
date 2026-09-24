import { nextTick } from 'vue'
import { useSettingsStore } from '@/stores/settings'
import { titleGenerator } from '@/lib/chat-title'
import { tileHost, tileTemplate } from '@/lib/maptiles'
import { savePreferences } from './preferences'
import { object } from './protocol'
import { studioSettingsLayout } from '@/lib/studio-settings-layout'

export function studioPreferences() {
  const s = useSettingsStore()
  return { maxTokens: s.maxTokens, maxToolCalls: s.maxToolCalls, summarize: s.summarize,
    autoTitle: s.autoTitle, markUnsure: s.markUnsure, mapTiles: s.mapTiles }
}
export function preferencePresentation() {
  const values = studioPreferences(), s = useSettingsStore()
  return { ...values, mapHost: tileHost(tileTemplate(values.mapTiles, s.theme)),
    layout: studioSettingsLayout() }
}
export function validatePreferences(value: unknown): Partial<ReturnType<typeof studioPreferences>> {
  const p = object(value)
  for (const [key, v] of Object.entries(p)) {
    if (key === 'maxTokens' || key === 'maxToolCalls') {
      if (v !== null && (!Number.isSafeInteger(v) || Number(v) < 1 || Number(v) > (key === 'maxTokens' ? 1048576 : 10000))) throw new Error('Invalid reply or tool-call limit')
    } else if (['summarize', 'autoTitle', 'markUnsure'].includes(key)) {
      if (typeof v !== 'boolean') throw new Error('Invalid preference switch')
    } else if (key === 'mapTiles') {
      if (typeof v !== 'string' || v.length > 4096) throw new Error('Invalid map address')
      if (v.trim()) {
        let url: URL
        try { url = new URL(v.replace(/\{[^}]*\}/g, '0')) } catch { throw new Error('Enter an absolute HTTP or HTTPS map address') }
        if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Map addresses must use HTTP(S), without embedded login credentials')
      }
    } else throw new Error('Unknown Studio preference')
  }
  return p
}

let queue: Promise<unknown> = Promise.resolve()
export function saveStudioPreferences(p: Record<string, unknown>) {
  const run = queue.catch(() => {}).then(async () => {
    const patch = validatePreferences(p.changes), expected = object(p.expected), before = studioPreferences()
    for (const key of Object.keys(patch) as (keyof typeof before)[]) {
      if (expected[key] !== before[key]) throw new Error('These preferences changed. Reload before applying; your draft is kept.')
    }
    const settings = useSettingsStore()
    settings.$patch(patch)
    try { await nextTick(); await savePreferences() }
    catch (e) {
      // Roll back only our fields. An unrelated UI preference is not ours to revert.
      const rollback = Object.fromEntries(Object.keys(patch).map(k => [k, before[k as keyof typeof before]]))
      settings.$patch(rollback); await nextTick(); throw e
    }
    if (patch.autoTitle === false) titleGenerator.cancel()
    return { preferences: preferencePresentation() }
  })
  queue = run
  return run
}
