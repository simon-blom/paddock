/** Only structured data crosses the app/content boundary. No JS snippets,
 * file paths, credentials, arbitrary URLs or generic core commands. */
export const VERSION = 1
export interface StoredAttachment { id: string; name: string; mime: string; size: number }
export interface AttachmentChoice { id: string; detail?: 'auto' | 'low' | 'high'; text?: boolean; from?: number; to?: number }
export interface NativeCommand { version: number; id: string; kind: string; payload: Record<string, unknown> }
const kinds = new Set(['refresh', 'newChat', 'open', 'models', 'settings', 'stage', 'removeAttachment', 'preview', 'openDocument', 'documentAction', 'closePreview', 'send', 'stop', 'quote', 'tools', 'composerSize', 'draft', 'samplerDefaults', 'renderer', 'shutdown'])
export function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected an object')
  return value as Record<string, unknown>
}
for (const kind of ['renameChat', 'pinChat', 'deleteChats', 'generateTitle', 'autoTitle', 'historyFilter', 'messageAction', 'toolPicker', 'toolQuery']) kinds.add(kind)
for (const kind of ['promptList', 'promptGet', 'promptSave', 'promptDelete', 'instructionsGet', 'instructionsApply', 'preferencesGet', 'preferencesSave']) kinds.add(kind)
for (const kind of ['microphoneStart', 'microphoneStop', 'microphoneCancel', 'microphoneSettings', 'microphoneRefresh', 'microphoneDevices', 'dictationAck']) kinds.add(kind)
kinds.add('transcriptExport')
kinds.add('transcriptMarks')
kinds.add('toolApproval')
kinds.add('graphPanel')
kinds.add('graphArtifact')
export function string(value: unknown, max = 1024): string {
  if (typeof value !== 'string' || value.length > max) throw new Error('Invalid text field')
  return value
}
export function identifier(value: unknown): string {
  const s = string(value, 128)
  if (!/^[a-zA-Z0-9_-]+$/.test(s)) throw new Error('Invalid identifier')
  return s
}
export function parseCommand(value: unknown): NativeCommand {
  const v = object(value)
  if (v.version !== VERSION || !kinds.has(String(v.kind)) || Object.keys(v).some(k => !['version', 'id', 'kind', 'payload'].includes(k))) throw new Error('Unsupported native content command')
  return { version: VERSION, id: identifier(v.id), kind: String(v.kind), payload: object(v.payload) }
}
export function storedAttachment(value: unknown): StoredAttachment {
  const v = object(value), id = identifier(v.id), name = string(v.name, 1024), mime = string(v.mime, 128)
  if (Object.keys(v).some(k => !['id', 'name', 'mime', 'size'].includes(k))) throw new Error('Unexpected attachment metadata')
  if (!name || /[\r\n]/.test(mime) || !Number.isSafeInteger(v.size) || Number(v.size) < 0 || Number(v.size) > 100 * 1024 * 1024) throw new Error('Invalid attachment metadata')
  return { id, name, mime, size: Number(v.size) }
}
export function attachmentChoices(value: unknown): AttachmentChoice[] {
  if (!Array.isArray(value) || value.length > 32) throw new Error('Invalid attachment list')
  const ids = new Set<string>()
  return value.map(entry => {
    const v = object(entry), id = identifier(v.id)
    if (ids.has(id)) throw new Error('Duplicate attachment')
    ids.add(id)
    if (v.detail !== undefined && !['auto', 'low', 'high'].includes(String(v.detail))) throw new Error('Invalid image detail')
    for (const key of ['from', 'to']) if (v[key] !== undefined && (!Number.isSafeInteger(v[key]) || Number(v[key]) < 1)) throw new Error('Invalid page range')
    if (v.from !== undefined && v.to !== undefined && Number(v.to) < Number(v.from)) throw new Error('Page range ends before it starts')
    if (v.text !== undefined && typeof v.text !== 'boolean') throw new Error('Invalid PDF reading mode')
    return { id, detail: v.detail as AttachmentChoice['detail'], text: v.text as boolean | undefined, from: v.from as number | undefined, to: v.to as number | undefined }
  })
}
