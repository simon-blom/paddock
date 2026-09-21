/** One recording owns its chunks until its asynchronous final data/stop events.
 * Capture teardown and another session never share or clear this buffer. */
export const NATIVE_RECORDING_TYPES = ['audio/mp4;codecs=mp4a.40.2', 'audio/mp4', 'audio/webm;codecs=opus', 'audio/webm'] as const
const WEB_TYPES = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']
const EXT: Record<string, string> = { webm: 'webm', mp4: 'm4a', ogg: 'ogg' }

export function recordAudio(stream: MediaStream, types: readonly string[] = WEB_TYPES, onSize?: (bytes: number) => void) {
  const mime = types.find(t => MediaRecorder.isTypeSupported(t))
  const recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
  let chunks: Blob[] = [], bytes = 0, cancelled = false, stopping = false
  let failure: Error | undefined
  let resolve!: (file: File | undefined) => void
  let reject!: (error: Error) => void
  const finished = new Promise<File | undefined>((yes, no) => { resolve = yes; reject = no })
  // Device errors may precede the caller's Stop. Keep the rejected result for
  // that caller, without an unhandled rejection while capture is active.
  void finished.catch(() => {})
  recorder.ondataavailable = e => {
    if (!cancelled && e.data.size) { chunks.push(e.data); bytes += e.data.size; onSize?.(bytes) }
  }
  recorder.onerror = () => { failure = new Error('The audio recorder failed before the recording was finalized') }
  recorder.onstop = () => {
    const type = recorder.mimeType || chunks.find(b => b.type)?.type || mime
    const parts = chunks; chunks = []
    recorder.ondataavailable = null; recorder.onstop = null; recorder.onerror = null
    if (cancelled) resolve(undefined)
    else if (failure) reject(failure)
    else if (!bytes || !type) resolve(undefined)
    else {
      const ext = EXT[type.split(';')[0]!.split('/')[1]!] ?? 'audio'
      resolve(new File(parts, `recording.${ext}`, { type }))
    }
  }
  recorder.start(1000)
  function stop() {
    if (stopping) return
    stopping = true
    // An inactive recorder can still have its final events queued.
    if (recorder.state !== 'inactive') recorder.stop()
  }
  return {
    finish() { stop(); return finished },
    cancel() { cancelled = true; chunks = []; resolve(undefined); stop() },
  }
}
export type AudioRecording = ReturnType<typeof recordAudio>
