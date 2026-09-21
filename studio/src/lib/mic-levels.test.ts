import { afterEach, expect, it, vi } from 'vitest'
import { createLevelMeter } from '../composables/useMicLevels'

afterEach(() => vi.unstubAllGlobals())

it('samples actual signal without animation frames and releases it on detach', () => {
  const raf = vi.fn(() => 1), cancel = vi.fn()
  vi.stubGlobal('requestAnimationFrame', raf)
  vi.stubGlobal('cancelAnimationFrame', cancel)
  let signal = 180
  const analyser = { frequencyBinCount: 512, disconnect: vi.fn(),
    getByteFrequencyData: (bins: Uint8Array) => bins.fill(signal) }
  const ac = { sampleRate: 16000, createAnalyser: () => analyser } as unknown as AudioContext
  const src = { connect: vi.fn() } as unknown as AudioNode
  const meter = createLevelMeter(9)
  meter.setFrameDriven(false)
  meter.attach(ac, src)
  expect(raf).not.toHaveBeenCalled()
  meter.sample()
  expect(meter.levels.value).toHaveLength(9)
  expect(meter.levels.value.every(n => n > 0 && n <= 1)).toBe(true)
  meter.detach()
  expect(analyser.disconnect).toHaveBeenCalledOnce()
  expect(meter.levels.value).toEqual([])
  signal = 0
  meter.attach(ac, src); meter.sample()
  expect(meter.levels.value).toEqual(Array(9).fill(0))
  meter.detach(); meter.sample()
  expect(meter.levels.value).toEqual([])
})

it('retains web frame-driven behavior and cancels only its own pending frame', () => {
  const raf = vi.fn(() => 17), cancel = vi.fn()
  vi.stubGlobal('requestAnimationFrame', raf)
  vi.stubGlobal('cancelAnimationFrame', cancel)
  const meter = createLevelMeter(9)
  const node = { frequencyBinCount: 512, disconnect: vi.fn() }
  meter.attach({ sampleRate: 48000, createAnalyser: () => node } as unknown as AudioContext,
    { connect: vi.fn() } as unknown as AudioNode)
  expect(raf).toHaveBeenCalledOnce()
  meter.setFrameDriven(false)
  expect(cancel).toHaveBeenCalledWith(17)
  meter.setFrameDriven(true)
  expect(raf).toHaveBeenCalledTimes(2)
  meter.detach()
  expect(cancel).toHaveBeenCalledTimes(2)
})
