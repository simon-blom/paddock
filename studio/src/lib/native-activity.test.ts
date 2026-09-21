import { describe, expect, it } from 'vitest'
import { desktopActivity, replyState } from '../../native-workspace/activity'
import { DEFAULT_PARAMS, type Conversation, type Message } from '@/types/chat'

function reply(id = 'answer'): Message {
  return { id, parentId: 'user', role: 'assistant', content: [{ type: 'text', text: 'PRIVATE ANSWER' }], reasoning: 'PRIVATE REASONING', createdAt: 0 }
}
function conversation(): Conversation {
  return { id: 'chat', title: 'PRIVATE TITLE', model: 'PRIVATE MODEL', systemPrompt: 'PRIVATE SYSTEM', params: { ...DEFAULT_PARAMS }, createdAt: 0, updatedAt: 0,
    leafId: 'answer', messages: [
      { id: 'user', parentId: null, role: 'user', content: [{ type: 'text', text: 'PRIVATE QUESTION' }], createdAt: 0 },
      { ...reply('hidden'), toolCalls: [{ id: 'call-hidden', approvalId: 'approval-hidden', status: 'pending', arguments: 'PRIVATE ARGUMENT', serverLabel: 'PRIVATE SERVER', name: 'PRIVATE TOOL' }] },
      { ...reply(), streaming: true, toolCalls: [{ id: 'call', approvalId: 'approval', status: 'pending', arguments: 'PRIVATE ARGUMENT', serverLabel: 'PRIVATE SERVER', name: 'PRIVATE TOOL' }] },
    ] }
}
describe('content-free desktop activity', () => {
  it('projects the active turn, never hidden branches or sensitive content', () => {
    const c = conversation(), before = JSON.stringify(c)
    const output = desktopActivity(c)
    expect(output).toEqual({ replies: [{ id: 'answer', state: 'streaming' }], approvals: ['approval'] })
    expect(JSON.stringify(output)).not.toContain('PRIVATE')
    expect(JSON.stringify(c)).toBe(before)
  })
  it('does not mistake cancellation, failure or truncation for completion', () => {
    expect(replyState(reply())).toBe('completed')
    expect(replyState({ ...reply(), stopped: true })).toBe('stopped')
    expect(replyState({ ...reply(), error: 'PRIVATE ERROR' })).toBe('failed')
    expect(replyState({ ...reply(), incomplete: 'length' })).toBe('incomplete')
    expect(replyState({ ...reply(), streaming: true })).toBe('streaming')
  })
  it('bounds pending approvals and ignores decided tools', () => {
    const c = conversation(), m = c.messages[2]
    m.toolCalls = Array.from({ length: 100 }, (_, i) => ({ id: `call-${i}`, approvalId: `approval-${i}`, status: i === 0 ? 'completed' : 'pending', arguments: 'PRIVATE', serverLabel: 'PRIVATE', name: 'PRIVATE' }))
    const a = desktopActivity(c)
    expect(a.approvals).toHaveLength(32)
    expect(a.approvals).not.toContain('approval-0')
    expect(desktopActivity(null)).toEqual({ replies: [], approvals: [] })
  })
})
