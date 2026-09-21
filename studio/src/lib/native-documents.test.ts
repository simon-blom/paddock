import { describe, expect, it } from 'vitest'
import { documentPreview, isNativeDocument, sentDocument } from '../../native-workspace/documents'
import { DEFAULT_PARAMS, type Conversation, type FilePart } from '@/types/chat'

const first: FilePart = { type: 'file', name: 'one.pdf', mime: 'application/pdf', attachmentId: 'first', pageRange: '2-4', pdfMode: 'text' }
const second: FilePart = { ...first, attachmentId: 'second', name: 'two.pdf' }
function fixture(): Conversation {
  return { id: 'c', title: 'Documents', model: 'fixture', systemPrompt: '', params: { ...DEFAULT_PARAMS }, createdAt: 0, updatedAt: 0,
    leafId: 'selected', activeDocId: 'u', messages: [
      { id: 'u', parentId: null, role: 'user', content: [first, second], createdAt: 0 },
      { id: 'hidden', parentId: 'u', role: 'user', content: [{ ...first, attachmentId: 'hidden-file' }], createdAt: 0 },
      { id: 'selected', parentId: 'u', role: 'assistant', content: [{ type: 'text', text: 'Reply' }], createdAt: 0 },
    ] }
}
describe('native document boundary', () => {
  it('selects the exact file in a multi-attachment turn without changing the conversation', () => {
    const c = fixture(), original = JSON.stringify(c)
    const one = sentDocument(c, 'u', 'first'), two = sentDocument(c, 'u', 'second')
    expect(one.conversation.id).not.toBe(two.conversation.id)
    expect(two.conversation.messages).toHaveLength(1)
    expect(two.conversation.messages[0].content).toEqual([second])
    expect(two.badge).toMatchObject({ id: 'second', name: 'two.pdf', pageRange: '2-4', textOnly: true })
    expect(JSON.stringify(c)).toBe(original)
  })
  it('rejects cross-branch, unknown-message and unknown-file references', () => {
    for (const [message, attachment] of [['hidden', 'hidden-file'], ['missing', 'first'], ['u', 'hidden-file']]) {
      expect(() => sentDocument(fixture(), message, attachment)).toThrow('not on this conversation branch')
    }
  })
  it('draft preview keeps only the chosen original and never creates a sent turn', () => {
    const c = fixture(), before = JSON.stringify(c)
    const preview = documentPreview(c, second, 'draft')
    expect(preview.conversation.messages[0].content).toEqual([second])
    expect(preview.conversation.id).not.toBe(c.id)
    expect(JSON.stringify(c)).toBe(before)
  })
  it('does not promote arbitrary paths, inline legacy images or unsupported formats', () => {
    expect(isNativeDocument({ ...first, attachmentId: '../secret' })).toBe(false)
    expect(isNativeDocument({ type: 'image', name: '', mime: '', attachmentId: '', dataUrl: 'data:image/png;base64,x' })).toBe(false)
    expect(isNativeDocument({ ...first, mime: 'application/zip', name: 'data.zip' })).toBe(false)
    expect(isNativeDocument({ ...first, mime: '', name: 'REPORT.PDF' })).toBe(true)
  })
})
