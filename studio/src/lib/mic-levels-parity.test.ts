import { afterEach, expect, it, vi } from 'vitest'
import { createLevelMeter } from '../composables/useMicLevels'
import fixtures from '../../../apps/macos/Tests/PaddockUITests/Fixtures/microphone-meter.json'

afterEach(() => vi.unstubAllGlobals())

for (const fixture of fixtures) {
  it(`native/web microphone spectrum contract: ${fixture.name}`, () => {
    let now = 1000
    vi.stubGlobal('performance', { now: () => now })
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    const bins = new Uint8Array(512)
    for (const signal of fixture.signal) bins.fill(signal.value, signal.start, signal.end)
    const node = { frequencyBinCount: 512, disconnect: vi.fn(),
      getByteFrequencyData: (destination: Uint8Array) => destination.set(bins) }
    const meter = createLevelMeter(9)
    meter.setFrameDriven(false)
    meter.attach({ sampleRate: fixture.sampleRate, createAnalyser: () => node } as unknown as AudioContext,
      { connect: vi.fn() } as unknown as AudioNode)
    for (let frame = 0; frame < fixture.frames; frame++) { meter.sample(); now += 16 }
    const elapsed = 1 / 60 + (fixture.frames - 1) * 0.016
    for (let band = 0; band < 9; band++) {
      const low = 80 * (6000 / 80) ** (band / 9)
      const high = 80 * (6000 / 80) ** ((band + 1) / 9)
      const start = Math.min(511, Math.max(0, Math.floor(low / (fixture.sampleRate / 2) * 512)))
      const end = Math.min(512, Math.max(start + 1, Math.ceil(high / (fixture.sampleRate / 2) * 512)))
      const mean = bins.slice(start, end).reduce((a, b) => a + b, 0) / (end - start) / 255
      const gain = Math.min(2.4, (Math.sqrt(low * high) / 80) ** 0.34)
      expect(meter.levels.value[band]).toBeCloseTo(Math.min(1, mean * gain) * (1 - Math.exp(-elapsed / 0.025)), 9)
    }
    meter.detach()
    expect(meter.levels.value).toEqual([])
  })
}
