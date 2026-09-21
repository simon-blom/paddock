import { describe, expect, it } from 'vitest'
import { DEFAULT_PARAMS, type Conversation, type Message } from '@/types/chat'
import { activeMessages, stepSibling } from './tree'
import { messageActions, resolveMessageAction } from '../../native-workspace/message-actions'
import { parseCommand } from '../../native-workspace/protocol'

function fixture(): Conversation {
  const m = (id: string, role: Message['role'], parentId: string | null): Message => ({ id, role, parentId, content: [{ type: 'text', text: id }], createdAt: 1 })
  return { id: 'fixture', title: 'Fixture', model: 'model', systemPrompt: '', params: { ...DEFAULT_PARAMS }, createdAt: 1, updatedAt: 1,
    leafId: 'b2', messages: [m('q', 'user', null), m('a', 'assistant', 'q'), m('q2', 'user', 'a'), m('a2', 'assistant', 'q2'), m('b2', 'assistant', 'q2'), m('b', 'assistant', 'q')] }
}
function request(c: Conversation, action: string, messageId: string, extra = {}) {
  return { action, conversationId: c.id, leafId: c.leafId, messageId, ...extra }
}
describe('native message action contracts', () => {
  it('exposes edit for historical questions, retry only at the tail, continue only for length', () => {
    const c = fixture()
    const controls = () => messageActions(c, activeMessages(c))
    expect(controls().get('q')).toMatchObject({ edit: true, retry: false, continueReply: false })
    expect(controls().get('a')?.retry).toBe(false)
    expect(controls().get('b2')).toMatchObject({ retry: true, continueReply: false, branch: { index: 2, count: 2, previous: 'a2', next: null } })
    c.messages[4].stopped = true
    expect(controls().get('b2')?.continueReply).toBe(false)
    c.messages[4].incomplete = 'length'
    expect(controls().get('b2')?.continueReply).toBe(true)
  })
  it('keeps the complete original attachment parts when editing text', () => {
    const c = fixture(), pdf = { type: 'file' as const, attachmentId: 'pdf', name: 'source.pdf', mime: 'application/pdf', pageRange: '2-4', pdfMode: 'text' as const }
    c.messages[0].content.push(pdf)
    const before = JSON.stringify(c)
    const action = resolveMessageAction(c, request(c, 'edit', 'q', { text: ' Edited question ', originalText: 'q' }))
    expect(action.parts).toEqual([pdf, { type: 'text', text: 'Edited question' }])
    expect(JSON.stringify(c)).toBe(before)
  })
  it('rejects stale conversations, leaves, off-path messages and original text', () => {
    const c = fixture()
    for (const fields of [{ conversationId: 'other' }, { leafId: 'a2' }, { messageId: 'b' }, { originalText: 'different' }]) {
      expect(() => resolveMessageAction(c, request(c, 'edit', 'q', { text: 'next', originalText: 'q', ...fields }))).toThrow()
    }
  })
  it('never accepts native replacement attachments, model identifiers or arbitrary tree parents', () => {
    const c = fixture()
    for (const field of ['parts', 'attachments', 'model', 'parentId', 'url']) {
      expect(() => resolveMessageAction(c, request(c, 'edit', 'q', { text: 'next', originalText: 'q', [field]: 'injected' }))).toThrow('Unexpected')
    }
    expect(parseCommand({ version: 1, id: 'command', kind: 'messageAction', payload: {} }).kind).toBe('messageAction')
  })
  it('selects exact adjacent siblings and remembers nested branch choices', () => {
    const c = fixture()
    const go = (source: string, targetId: string) => {
      const action = resolveMessageAction(c, request(c, 'branch', source, { targetId }))
      expect(stepSibling(c, source, action.delta)).toBe(true)
    }
    go('a', 'b'); expect(c.leafId).toBe('b')
    go('b', 'a'); expect(c.leafId).toBe('b2')
    go('b2', 'a2'); go('a', 'b'); go('b', 'a'); expect(c.leafId).toBe('a2')
    expect(() => resolveMessageAction(c, request(c, 'branch', 'a', { targetId: 'b2' }))).toThrow()
  })
  it('rejects invalid actions, empty edits, auto turns and streaming targets', () => {
    const c = fixture()
    for (const [action, id] of [['retry', 'a'], ['continue', 'b2'], ['delete', 'q'], ['edit', 'a']]) {
      expect(() => resolveMessageAction(c, request(c, action, id))).toThrow()
    }
    expect(() => resolveMessageAction(c, request(c, 'edit', 'q', { text: '  ', originalText: 'q' }))).toThrow()
    c.messages[0].auto = true
    expect(() => resolveMessageAction(c, request(c, 'edit', 'q', { text: 'next', originalText: 'q' }))).toThrow()
    c.messages[4].streaming = true
    expect(() => resolveMessageAction(c, request(c, 'retry', 'b2'))).toThrow()
  })
})
