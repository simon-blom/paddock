import { afterEach, expect, it, vi } from 'vitest'
import { recordAudio, NATIVE_RECORDING_TYPES } from './audio-recording'

class Recorder {
  static instances: Recorder[] = []
  static isTypeSupported = (type: string) => type.includes('webm') || type.includes('mp4')
  state = 'inactive'; mimeType: string
  ondataavailable: ((e: { data: Blob }) => void) | null = null
  onstop: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(_: unknown, opts?: { mimeType: string }) { this.mimeType = opts?.mimeType ?? 'audio/webm'; Recorder.instances.push(this) }
  start() { this.state = 'recording' }
  stop = vi.fn(() => { this.state = 'inactive' })
  data(value: string) { this.ondataavailable?.({ data: new Blob([value], { type: this.mimeType }) }) }
  end() { this.data('TAIL'); this.onstop?.() }
}
afterEach(() => { vi.unstubAllGlobals(); Recorder.instances = [] })
it('seals header, earlier chunks and final tail even after teardown or a new recording', async () => {
  vi.stubGlobal('MediaRecorder', Recorder)
  let active = recordAudio({} as MediaStream)
  const first = Recorder.instances[0]!
  first.data('HEADER'); first.data('BODY')
  const finalized = active.finish()
  expect(active.finish()).toBe(finalized)
  // Mirrors realtime: the socket settles before MediaRecorder's final events.
  active = recordAudio({} as MediaStream)
  const second = Recorder.instances[1]!
  second.data('NEW_HEADER')
  first.end()
  expect(await (await finalized)!.text()).toBe('HEADERBODYTAIL')
  const next = active.finish(); second.end()
  expect(await (await next)!.text()).toBe('NEW_HEADERTAIL')
  expect(first.stop).toHaveBeenCalledOnce()
})
it('discard releases a pending stop without later events contaminating another clip', async () => {
  vi.stubGlobal('MediaRecorder', Recorder)
  const clip = recordAudio({} as MediaStream)
  const r = Recorder.instances[0]!
  r.data('HEADER'); const done = clip.finish(); clip.cancel(); r.end()
  expect(await done).toBeUndefined()
  expect(r.stop).toHaveBeenCalledOnce()
})
it('native recording uses AAC/MP4 when supported and reports recorder failures', async () => {
  vi.stubGlobal('MediaRecorder', Recorder)
  const clip = recordAudio({} as MediaStream, NATIVE_RECORDING_TYPES)
  const r = Recorder.instances[0]!
  expect(r.mimeType).toBe('audio/mp4;codecs=mp4a.40.2')
  r.data('HEADER'); r.onerror?.()
  const done = clip.finish(); r.end()
  await expect(done).rejects.toThrow('finalized')
})
