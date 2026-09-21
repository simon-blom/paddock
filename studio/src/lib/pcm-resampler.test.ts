import { expect, it } from 'vitest'
import { PcmResampler } from './pcm-resampler'
it.each([16000, 24000, 44100, 48000, 96000])('preserves a second of %i Hz input across tiny audio quanta', rate => {
  const input = Float32Array.from({ length: rate }, (_, i) => Math.sin(i * 0.01))
  const whole = new PcmResampler(rate).push(input), stream = new PcmResampler(rate)
  const samples: number[] = []
  for (let i = 0; i < input.length; i += 128) samples.push(...stream.push(input.subarray(i, i + 128)))
  expect(samples).toHaveLength(16000)
  expect(whole.length).toBe(samples.length)
  expect(samples.every((v, i) => Math.abs(v - whole[i]) < 0.00001)).toBe(true)
})
it('does not lose fractional input at one-sample boundaries', () => {
  const stream = new PcmResampler(44100), samples: number[] = []
  for (let i = 0; i < 44100; i++) samples.push(...stream.push(new Float32Array([i / 44100])))
  expect(samples).toHaveLength(16000)
  expect(samples.every(Number.isFinite)).toBe(true)
})
