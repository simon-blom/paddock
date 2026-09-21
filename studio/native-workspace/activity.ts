import { activeSteps, stepMessages } from '@/lib/tree'
import type { Conversation, Message } from '@/types/chat'

// A content-free projection shared by web and native renderer modes. Never
// derive OS notifications from rendered text or export raw tool arguments.
export function replyState(m: Message) {
  return m.streaming ? 'streaming' : m.stopped ? 'stopped' : m.error ? 'failed' : m.incomplete ? 'incomplete' : 'completed'
}
export function desktopActivity(c: Conversation | null | undefined) {
  const steps = c ? activeSteps(c) : []
  const messages = steps.length ? stepMessages(steps[steps.length - 1]) : []
  const replies = messages.filter(m => m.role === 'assistant').slice(-4)
  return {
    replies: replies.map(m => ({ id: m.id, state: replyState(m) })),
    approvals: replies.flatMap(m => (m.toolCalls ?? []).filter(t => t.status === 'pending' && t.approvalId).map(t => t.approvalId!)).slice(0, 32),
  }
}
