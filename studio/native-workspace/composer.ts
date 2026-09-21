import { useModelsStore } from '@/stores/models'
import type { ContentPart, Conversation, SamplingParams, ToolSelection } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
import { rasterContext, isPdfPart } from '@/lib/docrun'
import { reasoningOptions, reasoningChoice, reasoningLabel, SAMPLER_DIALS, samplerIsSet, samplerLabel, samplerCaveats } from '@/lib/composer-policy'
import { audioPolicy } from '@/lib/audio-policy'

/** Bounded UI projection. Never sends messages, credentials, full thumbnails,
 * or draft text back to Swift. Model policy is the web composer's policy. */
export function composerPresentation(c: Conversation | null, staged: ContentPart[], toolGroups: { id: string; connectorId: string; tools: { name: string }[] }[]) {
  const models = useModelsStore(), id = c?.model ?? models.currentId
  const ids = c?.compareModels?.length ? c.compareModels : id && id !== 'default' ? [id] : []
  const p: SamplingParams = c?.params ?? DEFAULT_PARAMS
  const options = reasoningOptions(ids.map(id => models.reasoningLadderFor(id)))
  const choice = reasoningChoice(options, p, models.reasoningLadderFor(id).opens)
  const sampling = models.caps[id]?.sampling
  const { audioMode, audioOk } = audioPolicy(ids.map(id => ({ chat: models.canChat(id), audio: models.canTranscribe(id), live: false })), 0)
  const clips = staged.filter(p => p.type === 'audio')
  const limits = ids.map(id => models.caps[id]?.transcriptionMaxClipS).filter((s): s is number => typeof s === 'number' && s > 0)
  const docParser = !!models.caps[id]?.docParser
  const hasRaster = staged.some(p => p.type === 'image' || p.type === 'file' && isPdfPart(p)) || !!rasterContext(c)
  const hasImages = staged.some(p => p.type === 'image')
  const blind = ids.filter(id => !models.visionFor(id))
  const warnings: string[] = []
  let inputIssue = ''
  if (ids.length === 0 || ids.some(id => !models.models.some(m => m.id === id && m.status === 'ok'))) inputIssue = 'Select a reachable model to send.'
  else if (audioMode && !staged.some(p => p.type === 'audio')) inputIssue = 'Attach an audio clip for this model.'
  else if (clips.length && !audioOk) inputIssue = 'Every selected model must accept audio to send this clip.'
  else if (clips.length > 1) inputIssue = 'Attach one audio clip per turn.'
  else if (audioMode && staged.some(p => p.type !== 'audio')) inputIssue = 'Transcription accepts only an audio clip. Remove other attachments first.'
  else if (limits.length && clips.some(p => p.durationS != null && p.durationS > Math.min(...limits))) inputIssue = "This clip exceeds a selected model's audio limit. Choose a shorter clip or a model with a larger context."
  else if (hasImages && blind.length === ids.length) inputIssue = 'Select a vision model to send images.'
  else if (docParser && !hasRaster) inputIssue = 'Attach a page or image for this document model.'
  if (hasImages && blind.length && blind.length < ids.length) warnings.push(`${blind.map(id => models.models.find(m => m.id === id)?.display ?? id).join(', ')} cannot read images; those lanes will report an error.`)
  if (staged.some(p => p.type === 'image' && p.unreadable)) warnings.push('HEIC pictures are kept as files, but cannot be shown to the model. Convert to JPEG first.')
  // A configuration explanation belongs with Thinking/Compare controls, not
  // below Send. Keep actionable attachment warnings on the composer itself.
  const reasoningNotice = new Set(ids.map(id => models.reasoningStyleFor(id))).size > 1
    ? 'Models have different reasoning controls. Each lane uses the levels it supports.' : ''
  const selection: ToolSelection = c?.toolSelection ?? { mode: 'all' }
  const toolCount = toolGroups.reduce((sum, g) => sum + g.tools.filter(t => selection.mode === 'all'
    ? !g.connectorId || c?.connectorIds?.includes(g.connectorId)
    : selection.picks.some(p => p.label === g.id && (!p.tool || p.tool === t.name))).length, 0)
  return {
    reasoning: options.map(value => ({ value, label: reasoningLabel(value) })), reasoningChoice: choice,
    preserveThinking: ids.some(id => models.reasoningLadderFor(id).preserve),
    thinkingBudget: ids.some(id => models.thinkingBudgetFor(id)),
    webSearch: ids.some(id => models.webSearchFor(id)), audioMode, audioOk, docParser,
    inputIssue, warnings, reasoningNotice, toolCount, samplerSet: samplerIsSet(p),
    samplingSource: sampling?.source ?? '',
    samplingWarnings: samplerCaveats(ids.map(id => {
      const cloud = models.models.find(m => m.id === id)?.cloud
      return { id, kind: models.cloudEndpoints.find(e => e.id === cloud?.endpoint)?.kind }
    }), p).notes,
    sampling: SAMPLER_DIALS.map(d => {
      const advertised = d.wire in (sampling ?? {}) ? (sampling as unknown as Record<string, number>)[d.wire] : undefined
      return { key: d.key, label: d.label, min: d.min, max: d.max, step: d.step,
        value: p[d.key] ?? Math.min(d.max, Math.max(d.min, advertised ?? d.start)),
        display: samplerLabel(d.key, p[d.key], advertised), set: p[d.key] != null }
    }),
    cost: (c?.messages ?? []).reduce((sum, m) => sum + (m.usage?.costUsd ?? 0), 0),
  }
}
