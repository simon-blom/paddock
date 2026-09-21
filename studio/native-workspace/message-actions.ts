import { activeMessages, childIndex, siblingInfo, stepId, stepsOf } from '@/lib/tree'
import { messageText, type Conversation, type Message } from '@/types/chat'
import { identifier, string } from './protocol'

/** One index per projection, never one full tree walk per visible message. */
export function messageActions(c: Conversation, path: Message[]) {
  const siblings = new Map<string, { steps: ReturnType<typeof stepsOf>; index: number }>()
  for (const children of childIndex(c.messages).values()) {
    const steps = stepsOf(children)
    steps.forEach((s, index) => siblings.set(stepId(s), { steps, index }))
  }
  return new Map(path.map(m => {
    const { steps, index } = siblings.get(m.id) ?? { steps: [], index: -1 }
    const last = path[path.length - 1]?.id === m.id
    return [m.id, {
      edit: m.role === 'user' && !m.auto,
      // Web lanes are not individually retried/continued: a lane is part of
      // one group, not the tail answer of a single-model conversation.
      retry: last && m.role === 'assistant' && !m.group && !c.compareModels?.length,
      continueReply: last && m.role === 'assistant' && !m.group && !c.compareModels?.length && m.incomplete === 'length',
      branch: index >= 0 && steps.length > 1 ? {
        index: index + 1, count: steps.length,
        previous: index > 0 ? stepId(steps[index - 1]) : null,
        next: index + 1 < steps.length ? stepId(steps[index + 1]) : null,
      } : null,
    }]
  }))
}

/** Bind an action to the exact displayed path. Native never supplies content
 * parts, model URLs or tree links; attachments are retained from the store. */
export function resolveMessageAction(c: Conversation, p: Record<string, unknown>) {
  const action = string(p.action, 32)
  const fields = ['action', 'conversationId', 'leafId', 'messageId',
    ...(action === 'edit' ? ['text', 'originalText'] : action === 'branch' ? ['targetId'] : [])]
  if (Object.keys(p).some(k => !fields.includes(k))) throw new Error('Unexpected message action field')
  if (identifier(p.conversationId) !== c.id || identifier(p.leafId) !== c.leafId) throw new Error('The conversation branch changed. Try the action again.')
  const id = identifier(p.messageId), path = activeMessages(c), message = path.find(m => m.id === id)
  if (!message || message.streaming) throw new Error('This message is no longer available for this action')
  const controls = messageActions(c, path).get(id)!
  if (action === 'branch') {
    const targetId = identifier(p.targetId)
    const info = siblingInfo(c, id)!
    const delta = info.steps.findIndex(s => stepId(s) === targetId) - info.index
    if (!controls.branch || ![controls.branch.previous, controls.branch.next].includes(targetId) || Math.abs(delta) !== 1) throw new Error('That branch is no longer adjacent')
    return { action, message, delta, parts: [] }
  }
  if (action === 'edit' && controls.edit) {
    const text = string(p.text, 128 * 1024).trim()
    if (!text) throw new Error('Write a message before sending the edit')
    if (string(p.originalText, 128 * 1024) !== messageText(message)) throw new Error('The original message changed. Your edit has been kept.')
    return { action, message, delta: 0, parts: [...message.content.filter(p => p.type !== 'text'), { type: 'text' as const, text }] }
  }
  if ((action === 'retry' && controls.retry) || (action === 'continue' && controls.continueReply)) return { action, message, delta: 0, parts: [] }
  throw new Error('This action is not available for this message')
}
