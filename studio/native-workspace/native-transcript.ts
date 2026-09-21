import { activeSteps, stepId, stepMessages } from '@/lib/tree'
import { messageText, type Conversation } from '@/types/chat'
import { messageStamp } from '@/lib/model-name'
import { answerMetrics, answerMetricsHint, runDetailSections, thinkingLabel, thinkingMetrics, tokenLimitNote } from '@/lib/message-presentation'
import { documentBadge, isNativeDocument } from './documents'
import { messageActions } from './message-actions'
import { isNativeAudio, audioBadge, speechProjection } from './speech'
import { messageExtras } from './message-extras'

/** Native presentation of every active tree step. No raw response objects,
 * attachment bytes or credentials. Large states use framed transport, never
 * a renderer switch or truncated conversation. */
export function nativeTranscript(c: Conversation | null | undefined) {
  const steps = c ? activeSteps(c) : []
  const messages = steps.flatMap(stepMessages)
  const groups = new Map(steps.flatMap(s => s.kind === 'group' ? s.ms.map(m => [m.id, stepId(s)] as const) : []))
  const actions = c ? messageActions(c, messages) : new Map()
  const speech = speechProjection(messages)
  const projected = messages.map(m => {
    const stamp = messageStamp(m), text = messageText(m)
    const thinking = !!m.streaming && !text
    return { id: m.id, role: m.role, text, group: groups.get(m.id) ?? null, ...messageExtras(m),
      contended: !!m.run?.contended, actions: actions.get(m.id), attachments: m.content.filter(isNativeDocument).map(documentBadge),
      audioClips: m.content.filter(isNativeAudio).map(audioBadge), speech: speech.get(m.id) ?? null,
      audioPending: m.content.some(p => p.type === 'audio' && p.attachmentId === ''),
      reasoning: m.reasoning ?? '', model: stamp.id, streaming: !!m.streaming,
      stopped: !!m.stopped, error: m.error ?? '', incomplete: !!m.incomplete,
      chrome: m.role === 'assistant' ? {
        modelName: stamp.label, vendor: stamp.vendor, spec: stamp.spec,
        footer: answerMetrics(m.usage), footerHint: answerMetricsHint(m.usage),
        thinkingLabel: thinkingLabel(m.usage?.reasoningMs, thinking),
        thinkingMeta: thinkingMetrics(m.usage?.reasoningTokens, m.usage?.reasoningTps, thinking),
        cutNote: tokenLimitNote(m.usage), sections: runDetailSections(m),
        promptText: m.run?.systemPrompt ?? '',
      } : null,
    }
  })
  return { available: true, notice: '', conversationId: c?.id ?? null, leafId: c?.leafId ?? null, messages: projected }
}
