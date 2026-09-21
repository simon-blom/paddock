import { activeMessages } from '@/lib/tree'
import { isDocxPart, isPdfPart } from '@/lib/docrun'
import type { ContentPart, Conversation, FilePart, ImagePart } from '@/types/chat'

/** Stored originals only. Legacy inline images and unsupported formats keep
 * their shared Studio interactions; never put bytes or arbitrary URLs on the
 * native presentation bridge. */
export function isNativeDocument(p: ContentPart): p is FilePart | ImagePart {
  return (p.type === 'image' || (p.type === 'file' && (isPdfPart(p) || isDocxPart(p))))
    && /^[a-zA-Z0-9_-]{1,128}$/.test(p.attachmentId)
}
export function documentBadge(p: FilePart | ImagePart) {
  return { id: p.attachmentId, name: p.name, kind: p.type === 'image' ? 'image' : isPdfPart(p) ? 'pdf' : 'docx',
    pages: p.type === 'file' ? p.pages : undefined, pageRange: p.pageRange,
    textOnly: p.type === 'file' && p.pdfMode === 'text' }
}

/** A viewer-only projection, never persisted or passed to inference. Resolving
 * one part avoids docContext's historical one-document-per-user-turn grouping.
 * The original message, page range and activeDocId are not modified. */
export function documentPreview(c: Conversation, p: ContentPart, source: string) {
  if (!isNativeDocument(p)) throw new Error('This attachment needs the shared Studio viewer')
  const id = `document-${source}-${p.attachmentId}`
  return { badge: documentBadge(p), conversation: { ...c, id, leafId: id, activeDocId: id,
    messages: [{ id, role: 'user' as const, parentId: null, content: [p], createdAt: 0 }] } }
}

export function sentDocument(c: Conversation, messageId: string, attachmentId: string) {
  const m = activeMessages(c).find(m => m.id === messageId && m.role === 'user')
  const p = m?.content.find(p => isNativeDocument(p) && p.attachmentId === attachmentId)
  if (!p) throw new Error('The document is not on this conversation branch')
  return documentPreview(c, p, messageId)
}
