/** Streaming fallback when a browser cannot create a 16 kHz AudioContext.
 * Preserve fractional phase across render quanta; rounding each 128-sample
 * block separately shortens 48 kHz audio by 1.56% and shifts word times. */
export class PcmResampler {
  private offset = 0
  private next = 0
  private previous = 0
  constructor(private readonly from: number, private readonly to = 16000) {
    if (!(from > 0 && to > 0 && Number.isFinite(from) && Number.isFinite(to))) throw new Error('Invalid audio sample rate')
  }
  push(input: Float32Array): Float32Array {
    if (!input.length) return new Float32Array()
    if (this.from === this.to) return input
    const ratio = this.from / this.to, end = this.offset + input.length - 1
    const out = new Float32Array(Math.max(0, Math.floor((end - this.next) / ratio) + 1))
    for (let i = 0; i < out.length; i++, this.next += ratio) {
      const low = Math.floor(this.next), fraction = this.next - low
      const a = low < this.offset ? this.previous : input[low - this.offset]
      const b = input[Math.min(low + 1 - this.offset, input.length - 1)]
      out[i] = a + (b - a) * fraction
    }
    this.previous = input[input.length - 1]; this.offset += input.length
    return out
  }
}
