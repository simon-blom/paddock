import { describe, expect, it } from 'vitest'
import mainSource from '../../native-workspace/main.ts?raw'
import viewerSource from '../../native-workspace/EmbeddedViewers.vue?raw'
import buildSource from '../../native-workspace/vite.config.ts?raw'
import transcriptSource from '../../native-workspace/native-transcript.ts?raw'
import { presentationFrames } from '../../native-workspace/presentation-frames'
import { messageExtras, pendingApproval } from '../../native-workspace/message-extras'
import { DEFAULT_PARAMS, type Conversation, type Message } from '@/types/chat'

const answer: Message = { id: 'a', parentId: 'u', role: 'assistant', createdAt: 1, content: [],
  toolCalls: [{ id: 'call', name: 'write_file', serverLabel: 'files', arguments: '{"path":"notes.txt"}', status: 'pending', approvalId: 'gate' }] }
const conversation: Conversation = { id: 'c', title: 'Fixture', model: 'm', createdAt: 1, updatedAt: 1,
  systemPrompt: '', params: { ...DEFAULT_PARAMS }, leafId: 'a', messages: [
    { id: 'u', parentId: null, role: 'user', createdAt: 1, content: [{ type: 'text', text: 'Test' }] }, answer] }
describe('native-only rendering boundary', () => {
  it('frames Unicode/control-heavy state losslessly below each bridge limit', () => {
    const value = { revision: 7, text: '😀🦊\u0000\n<script>'.repeat(40000) }
    const frames = presentationFrames(value) as { transfer: number; index: number; count: number; payload: string }[]
    expect(frames.length).toBeGreaterThan(1)
    expect(frames.every((f, i) => f.index === i && f.transfer === 7 && f.count === frames.length)).toBe(true)
    expect(frames.every(f => new TextEncoder().encode(JSON.stringify(f)).length < 256 * 1024)).toBe(true)
    const bytes = Uint8Array.from(frames.map(f => atob(f.payload)).join(''), c => c.charCodeAt(0))
    expect(JSON.parse(new TextDecoder().decode(bytes))).toEqual(value)
    expect(() => presentationFrames(value, 10)).toThrow('safety limit')
  })
  it('retains approvals and rejects every stale/foreign/settled decision', () => {
    const p = { conversationId: 'c', leafId: 'a', messageId: 'a', callId: 'call', approvalId: 'gate', approve: true }
    expect(pendingApproval(conversation, p)).toEqual({ id: 'gate', approve: true })
    expect(pendingApproval(conversation, { ...p, approve: false }).approve).toBe(false)
    for (const patch of [{ conversationId: 'other' }, { leafId: 'other' }, { messageId: 'u' }, { callId: 'other' }, { approvalId: 'old' }, { approve: 'true' }]) {
      expect(() => pendingApproval(conversation, { ...p, ...patch })).toThrow()
    }
    expect(() => pendingApproval({ ...conversation, messages: [conversation.messages[0], { ...answer, toolCalls: [{ ...answer.toolCalls![0], status: 'completed' }] }] }, p)).toThrow()
    expect(messageExtras(answer).toolCalls[0]).toMatchObject({ status: 'pending', approvalId: 'gate', arguments: '{"path":"notes.txt"}' })
  })
  it('projects OCR pages, confidence, search and automatic steps without raw responses', () => {
    const extra = messageExtras({ ...answer, auto: true,
      docRun: { pages: [{ state: 'review', text: 'Visible text', note: 'Check this page', words: [{ w: 'Visible', c: .2 }], regions: [{ label: 'text', boxes: [[1, 2, 3, 4]] }] }] },
      webSearches: [{ id: 's', query: 'test', status: 'failed', sources: [{ title: 'source', url: 'https://example.com' }], error: 'Search timed out', provider: 'brave' }] })
    expect(extra.automatic).toBe(true)
    expect(extra.documentResult?.pages[0]).toMatchObject({ id: 1, text: 'Visible text', note: 'Check this page', unsure: [{ label: 'Visible', value: '20%' }] })
    expect(extra.searches[0]).toMatchObject({ error: 'Search timed out', sources: [{ title: 'source', url: 'https://example.com' }] })
  })
  it('entry point mounts only the embedded-viewer root and build rejects web app controls', () => {
    expect(mainSource).not.toMatch(/ChatView|ChatThread|Toaster|Composer/)
    expect(viewerSource).not.toMatch(/import\(['"].*(?:ChatView|ChatThread|MessageBubble|AudioPlayer|ArtifactPanel)/)
    expect(buildSource).toContain('Web application UI entered the native bundle')
    expect(transcriptSource).not.toMatch(/available: false|128 \* 1024|unsupported/)
  })
})
