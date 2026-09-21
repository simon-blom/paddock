import { usePromptsStore } from '@/stores/prompts'
import type { SavedPrompt } from '@/lib/api'
import { identifier, object, string } from './protocol'

export function promptPage(rows: SavedPrompt[], value: unknown) {
  const p = object(value), search = string(p.search ?? '', 512).trim().toLowerCase()
  if (!Number.isSafeInteger(p.page ?? 0) || Number(p.page ?? 0) < 0) throw new Error('Invalid preset page')
  const matches = rows.filter(r => `${r.name}\n${r.body}`.toLowerCase().includes(search))
  const pageSize = 40, page = Math.min(Number(p.page ?? 0), Math.max(0, Math.ceil(matches.length / pageSize) - 1))
  return { page, pageSize, total: rows.length, matched: matches.length,
    rows: matches.slice(page * pageSize, (page + 1) * pageSize).map(r => ({ id: r.id, name: r.name,
      preview: r.body.slice(0, 256), revision: r.revision ?? '', updatedAt: r.updatedAt ?? 0 })) }
}

export function presetText(value: unknown, max: number): string {
  const result = string(value, max)
  if (new TextEncoder().encode(result).length > max) throw new Error('Preset text exceeds its byte limit')
  return result
}

/** Library bodies are requested only when opened, never repeated on token
 * updates. Search runs over the full shared library before bounded paging. */
export async function promptCommand(kind: string, p: Record<string, unknown>) {
  const store = usePromptsStore()
  if (kind === 'promptList' || kind === 'promptGet') {
    await store.refresh()
    if (store.error) throw new Error(store.error)
    if (kind === 'promptList') return { library: promptPage(store.prompts, p) }
    const record = store.prompts.find(r => r.id === identifier(p.id))
    if (!record) throw new Error('This preset no longer exists')
    presetText(record.body, 128 * 1024)
    return { prompt: record }
  }
  const id = identifier(p.id), revision = string(p.revision, 64)
  if (kind === 'promptSave') {
    const name = presetText(p.name, 512).trim(), body = presetText(p.body, 128 * 1024)
    if (!name || !body.trim()) throw new Error('Enter a name and prompt text')
    return { prompt: await store.save(name, body, id, revision) }
  }
  if (kind === 'promptDelete') { await store.remove(id, revision); return { deleted: id } }
  throw new Error('Unknown preset command')
}
