import { activeMessages } from '@/lib/tree'
import { renderWords, disagreements } from '@/lib/transcript-diff'
import { languageName } from '@/lib/languages'
import { clock, srt, vtt } from '@/lib/subtitles'
import { messageText, type AudioPart, type ContentPart, type Conversation, type Message } from '@/types/chat'

export function isNativeAudio(p: ContentPart): p is AudioPart {
  return p.type === 'audio' && /^[a-zA-Z0-9_-]{1,128}$/.test(p.attachmentId)
}
export function audioBadge(p: AudioPart) {
  return { id: p.attachmentId, name: p.name || 'Recording', mime: p.mime, size: p.size, duration: p.durationS }
}
export function speechProjection(messages: Message[]) {
  const result = new Map<string, ReturnType<typeof project>>()
  let clip: AudioPart | undefined
  const groups = new Map<string, Message[]>()
  for (const m of messages) {
    if (m.role === 'user') clip = m.content.find(isNativeAudio)
    if (!m.transcript) continue
    result.set(m.id, project(m, clip))
    if (m.group) groups.set(m.group, [...(groups.get(m.group) ?? []), m])
  }
  for (const lanes of groups.values()) {
    const marks = disagreements(lanes.map(m => result.get(m.id)!.words.map(w => w.word)))
    lanes.forEach((m, i) => { result.get(m.id)!.differs = [...marks[i]!] })
  }
  return result
}
function project(m: Message, clip?: AudioPart) {
  const t = m.transcript!
  const seconds = clip?.durationS ?? t.durationS
  const facts: { label: string; value: string }[] = []
  if (t.language) facts.push({ label: clip?.language ? 'Language' : 'Detected', value: languageName(t.language) })
  if (seconds) facts.push({ label: 'Audio', value: clock(seconds) })
  if (seconds && m.usage?.ms && m.usage.ms > 0) facts.push({ label: 'Speed', value: `${(seconds * 1000 / m.usage.ms).toFixed(1)}× realtime` })
  if (t.wordsFrom) facts.push({ label: 'Word timing', value: t.wordsFrom + (t.wordsLangOk === false ? ' · outside its languages' : '') })
  // Only rendering fields, never raw response objects or arbitrary metadata.
  return { clip: clip ? audioBadge(clip) : null,
    words: renderWords(t.segments, t.words, messageText(m)), differs: [] as number[], facts,
    guards: (t.guards ?? []).map(g => ({ start: g.start, end: g.end, note: g.note })),
    subtitleExport: !!t.segments?.length,
  }
}

export function exportTranscript(c: Conversation, p: Record<string, unknown>) {
  if (p.conversationId !== c.id || p.leafId !== c.leafId) throw new Error('The conversation branch changed')
  const m = activeMessages(c).find(m => m.id === p.messageId)
  if (!m?.transcript || m.streaming) throw new Error('Choose a completed transcription')
  const t = m.transcript, text = messageText(m), cues = t.segments ?? []
  const kind = p.format
  if (!['txt', 'json', 'srt', 'vtt'].includes(String(kind))) throw new Error('Unknown export format')
  if ((kind === 'srt' || kind === 'vtt') && !cues.length) throw new Error('This model did not supply subtitle times')
  return { name: `transcript.${kind}`, text: kind === 'txt' ? text : kind === 'srt' ? srt(cues) : kind === 'vtt' ? vtt(cues)
    : JSON.stringify({ text, language: t.language, duration: t.durationS, segments: t.segments, words: t.words, guards: t.guards }, null, 2) }
}
