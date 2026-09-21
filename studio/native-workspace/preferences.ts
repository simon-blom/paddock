// The private host has a fresh origin and a nonpersistent WebKit store. Keep
// only explicit UI preferences in the existing product SQLite, not WebKit
// cookies, caches, secrets, microphone grants or arbitrary localStorage keys.
const keys = new Set([
  'pk_theme', 'pk_max_tokens', 'pk_max_tokens_v2', 'pk_max_tool_calls', 'pk_summarize', 'pk_auto_title', 'pk_mark_unsure',
  'pk_dictate_with', 'pk_mic_device', 'pk_mic_device_label', 'pk_map_tiles',
  'pk_sidebar_width', 'pk_sidebar_open_width', 'pk_artifact_width_v2',
  'pk_graphpane_width_v1', 'pk_docpane_width_v2', 'pk_gpu_dock_width',
])
const slot = 'macos_studio_preferences'
let saved = '', pending: Promise<void> | undefined
function snapshot(): Record<string, string> {
  return Object.fromEntries([...keys].flatMap(key => {
    const value = localStorage.getItem(key)
    return value !== null && value.length <= 8192 ? [[key, value]] : []
  }))
}
export async function restorePreferences(): Promise<void> {
  const response = await fetch('/api/settings')
  if (!response.ok) throw new Error('Studio preferences could not be opened')
  const data = (await response.json())[slot]
  if (data && typeof data === 'object') for (const [key, value] of Object.entries(data)) {
    if (keys.has(key) && typeof value === 'string' && value.length <= 8192) localStorage.setItem(key, value)
  }
  // Native snapshots were written after the web migration ran. Older native
  // builds omitted its marker; an explicitly chosen 8192 is not the old default.
  if (data && typeof data.pk_max_tokens === 'string') localStorage.setItem('pk_max_tokens_v2', '1')
  saved = JSON.stringify(snapshot())
}
export async function savePreferences(): Promise<void> {
  if (pending) await pending
  const data = snapshot(), serialized = JSON.stringify(data)
  if (serialized === saved) return
  pending = (async () => {
    const response = await fetch('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ [slot]: data }) })
    if (!response.ok) throw new Error('Studio preferences could not be saved')
    saved = serialized
  })()
  try { await pending } finally { pending = undefined }
}
