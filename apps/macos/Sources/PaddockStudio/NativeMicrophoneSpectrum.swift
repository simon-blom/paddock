import Accelerate
import Foundation

/// Owned solely by the capture sample queue. Same immutable PCM as the WAV and
/// transcriber, not a second input or an audio gain/normalisation stage. The
/// fixed ring, transform and scratch buffers are reused for the whole session.
final class NativeMicrophoneSpectrum {
  static let size = 1024
  static let hop = 256
  private let transform: vDSP.DiscreteFourierTransform<Float>?
  private let window: [Float]
  private var ring = [Float](repeating: 0, count: size)
  private var real = [Float](repeating: 0, count: size)
  private let imaginary = [Float](repeating: 0, count: size)
  private var outputReal = [Float](repeating: 0, count: size)
  private var outputImaginary = [Float](repeating: 0, count: size)
  private var magnitude = [Float](repeating: 0, count: size / 2)
  private var bins = [UInt8](repeating: 0, count: size / 2)
  private var writeIndex = 0, sinceTransform = 0
  private var shape = MicrophoneSpectrumShape()
  var levels: [Double] { shape.levels }

  init() {
    transform = try? vDSP.DiscreteFourierTransform(
      count: Self.size, direction: .forward, transformType: .complexComplex, ofType: Float.self)
    // Periodic Blackman window, as used by the web analyser.
    window = (0..<Self.size).map {
      let angle = 2 * Double.pi * Double($0) / Double(Self.size)
      return Float(0.42 - 0.5 * cos(angle) + 0.08 * cos(2 * angle))
    }
  }

  @discardableResult func consume(_ pcm: Data) -> Bool {
    guard transform != nil else { return false }  // Meter failure cannot stop recording.
    var updated = false
    pcm.withUnsafeBytes { raw in
      for offset in stride(from: 0, to: pcm.count - pcm.count % 2, by: 2) {
        let sample = Int16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: Int16.self))
        ring[writeIndex] = Float(sample) / 32768
        writeIndex = (writeIndex + 1) & (Self.size - 1)
        sinceTransform += 1
        if sinceTransform == Self.hop {
          analyse()
          sinceTransform = 0
          updated = true
        }
      }
    }
    return updated
  }

  private func analyse() {
    for i in 0..<Self.size { real[i] = ring[(writeIndex + i) & (Self.size - 1)] * window[i] }
    transform?.transform(
      inputReal: real, inputImaginary: imaginary,
      outputReal: &outputReal, outputImaginary: &outputImaginary)
    for i in bins.indices {
      // Complex DFT is unnormalised. Divide by N before dB conversion, matching
      // the web analyser's frequency magnitudes (not an RMS amplitude scale).
      let value = hypot(outputReal[i], outputImaginary[i]) / Float(Self.size)
      magnitude[i] = 0.3 * magnitude[i] + 0.7 * value
      let db = 20 * log10(max(Double(magnitude[i]), 1e-12))
      let normalized =
        (db - MicrophoneSpectrumShape.minimumDB)
        / (MicrophoneSpectrumShape.maximumDB - MicrophoneSpectrumShape.minimumDB)
      bins[i] = UInt8(min(255, max(0, floor(normalized * 255))))
    }
    shape.update(bins: bins, elapsed: Double(Self.hop) / 16000)
  }
}
