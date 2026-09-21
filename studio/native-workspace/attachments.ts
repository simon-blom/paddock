import { attachmentsApi } from '@/lib/api'
import { isPdfPart } from '@/lib/docrun'
import { pdfPageCount } from '@/lib/pdf'
import type { ContentPart, ImagePart } from '@/types/chat'
import type { StoredAttachment } from './protocol'
import { audioDuration, isAudioFile } from '@/lib/transcribe'

/** Swift already streamed the original to Rust. Build display metadata from
 * its scoped attachment URL; never re-upload, inline the original, or expose
 * its source path. Large PDF bytes are released after worker page counting. */
export async function describeAttachment(a: StoredAttachment): Promise<ContentPart> {
  if (/\.tvdb$/i.test(a.name)) return { type: 'graph', attachmentId: a.id, name: a.name, size: a.size }
  if (a.mime.startsWith('image/')) {
    const part: ImagePart = { type: 'image', attachmentId: a.id, name: a.name, mime: a.mime, detail: 'auto' }
    if (/\.hei[cf]$/i.test(a.name) || /^image\/hei[cf]$/.test(a.mime)) part.unreadable = true
    try {
      const image = new Image()
      image.src = attachmentsApi.url(a.id)
      await image.decode()
      part.width = image.naturalWidth; part.height = image.naturalHeight
      const scale = Math.min(1, 160 / Math.max(image.naturalWidth, image.naturalHeight))
      const canvas = document.createElement('canvas')
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale))
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale))
      canvas.getContext('2d')?.drawImage(image, 0, 0, canvas.width, canvas.height)
      part.thumbUrl = canvas.toDataURL('image/jpeg', .7)
      canvas.width = canvas.height = 0
      image.src = ''
    } catch { /* Unsupported image previews remain named, downloadable cards. */ }
    return part
  }
  if (isAudioFile({ type: a.mime, name: a.name })) {
    const durationS = await audioDuration(attachmentsApi.url(a.id))
    return { type: 'audio', attachmentId: a.id, name: a.name, mime: a.mime, size: a.size, durationS }
  }
  let pages: number | undefined
  if (isPdfPart(a)) {
    const response = await fetch(attachmentsApi.url(a.id))
    if (!response.ok) throw new Error('The uploaded document could not be opened')
    try { pages = await pdfPageCount(await response.arrayBuffer()) || undefined }
    catch { /* A malformed PDF remains attachable, with explicit viewer/server errors. */ }
  }
  return { type: 'file', attachmentId: a.id, name: a.name, mime: a.mime, size: a.size, pages }
}
