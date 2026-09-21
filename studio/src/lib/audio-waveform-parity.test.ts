import { afterEach, expect, it, vi } from 'vitest'
import { useAudioPeaks } from '../composables/useAudioPeaks'
import fixtures from '../../../apps/macos/Tests/PaddockUITests/Fixtures/audio-waveform.json'

afterEach(() => vi.unstubAllGlobals())
for (const fixture of fixtures) {
  it(`native/web waveform peaks: ${fixture.name}`, async () => {
    const data = fixture.channels.map(c => Float32Array.from(c))
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, headers: new Headers(), arrayBuffer: async () => new ArrayBuffer(8) })))
    const close = vi.fn(async () => {})
    class Context {
      close = close
      async decodeAudioData() { return { duration: data[0]!.length / 16000, length: data[0]!.length,
        numberOfChannels: data.length, getChannelData: (channel: number) => data[channel] } }
    }
    vi.stubGlobal('window', { AudioContext: Context })
    const state = useAudioPeaks()
    await state.load(fixture.name, 'https://fixture.invalid/audio')
    const peaks = state.peaks.value!
    expect(peaks).not.toBeNull()
    for (let bucket = 0; bucket < 2048; bucket++) {
      const index = Math.floor(bucket * data[0]!.length / 2048)
      expect(peaks.min[bucket]).toBe(fixture.minimum[index])
      expect(peaks.max[bucket]).toBe(fixture.maximum[index])
    }
    expect(close).toHaveBeenCalledOnce()
  })
}
