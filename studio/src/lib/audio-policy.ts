/** One capability policy for the web composer and its native presentation. */
export type MicMode = 'live' | 'record' | 'dictate'
export interface AudioLane { chat: boolean; audio: boolean; live: boolean }
export interface RealtimeTranscriptionCaps { supported: boolean; enrichment: boolean; drain?: boolean; final_revision?: boolean; utterance_max_s?: number | null }
/** Explicit live capability wins. Older Whisper runners need both timestamp
 * mechanisms; Granite Plus's file-only word instruction is not live metadata. */
export function realtimeEnrichment(caps?: RealtimeTranscriptionCaps, times: readonly string[] = []): boolean {
  return caps ? caps.supported && caps.enrichment : times.includes('segment') && times.includes('word')
}
export function audioPolicy(lanes: AudioLane[], transcribers: number, docParser = false) {
  const audioOk = lanes.length > 0 && lanes.every(l => l.audio)
  const audioMode = audioOk && !lanes.every(l => l.chat)
  const liveBlocked = audioMode && lanes.some(l => !l.live)
  const jobs: MicMode[] = docParser ? [] : !audioOk ? (transcribers ? ['dictate'] : [])
    : audioMode ? (liveBlocked ? ['record'] : ['live', 'record'])
      : lanes.length > 1 ? ['record'] : transcribers ? ['record', 'dictate'] : ['record']
  return { audioOk, audioMode, liveBlocked, jobs }
}
