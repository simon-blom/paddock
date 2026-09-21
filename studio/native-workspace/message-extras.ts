import type { Conversation, Message } from '@/types/chat'
import { activeMessages } from '@/lib/tree'
import { cleanOcrText, htmlTablesToMarkdown } from '@/lib/ocr'
import { isNativeAudio } from './speech'
import { isNativeDocument } from './documents'
import { identifier } from './protocol'

/** Allowlisted display data only; tool output stays inert text in TextKit. */
export function messageExtras(m: Message) {
  return {
    automatic: !!m.auto,
    toolCalls: (m.toolCalls ?? []).map(t => ({ id: t.id, name: t.name, server: t.serverLabel,
      arguments: t.arguments ?? '', output: t.output ?? '', status: t.status,
      error: typeof t.error === 'string' ? t.error : t.error ? 'Tool failed' : '',
      approvalId: t.approvalId ?? null,
      artifactId: /^artifacts__artifact_|^artifact_/.test(t.name ?? '')
        ? `${t.arguments} ${t.output}`.match(/art_[0-9a-f]{12}/)?.[0] ?? null : null })),
    searches: (m.webSearches ?? []).map(s => ({ id: s.id, query: s.query, status: s.status,
      provider: s.provider ?? '', error: s.error ?? '',
      sources: s.sources.map(v => ({ title: v.title ?? v.url, url: v.url })) })),
    files: m.content.flatMap(p => p.type !== 'text' && !(p.type === 'audio' && !p.attachmentId)
      && !isNativeDocument(p) && !isNativeAudio(p)
      ? [{ id: p.attachmentId || `legacy-${m.id}`, name: p.name || 'Attachment',
        kind: p.type, mime: 'mime' in p ? p.mime : '', stored: /^[a-zA-Z0-9_-]{1,128}$/.test(p.attachmentId) }] : []),
    documentResult: m.ocr || m.docRun ? {
      facts: Object.entries(m.ocr ?? {}).flatMap(([label, value]) =>
        typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
          ? [{ label, value: String(value) }] : []),
      pages: (m.docRun?.pages ?? [{ state: m.streaming ? 'reading' : 'done',
        text: m.content.filter(p => p.type === 'text').map(p => p.text).join('\n'), regions: m.ocr?.regions }])
        .map((p, i) => ({ id: i + 1, state: p.state, text: htmlTablesToMarkdown(cleanOcrText(p.text)),
          note: p.note ?? '', regions: (p.regions ?? []).map(r => ({ label: r.label, text: r.text ?? '', boxes: r.boxes, quads: r.quads ?? [] })),
          unsure: (p.words ?? []).filter(w => w.c < .45).map(w => ({ label: w.w, value: `${Math.round(w.c * 100)}%` })) }))
    } : null,
  }
}

/** Approval targets are re-resolved at click time, never trusted from a stale
 * native card or allowed to approve another branch/conversation. */
export function pendingApproval(c: Conversation, p: Record<string, unknown>) {
  if (p.conversationId !== c.id || p.leafId !== (c.leafId ?? null) || typeof p.approve !== 'boolean') {
    throw new Error('The conversation changed; review the current tool request')
  }
  const m = activeMessages(c).find(m => m.id === identifier(p.messageId))
  const call = m?.toolCalls?.find(t => t.id === identifier(p.callId))
  if (!call || call.status !== 'pending' || !call.approvalId || call.approvalId !== p.approvalId) {
    throw new Error('This tool request is no longer awaiting approval')
  }
  return { id: call.approvalId, approve: p.approve }
}
