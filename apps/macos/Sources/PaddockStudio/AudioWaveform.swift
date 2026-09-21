import AVFoundation
import Accelerate
import Foundation

/// A bounded, exact min/max envelope. Every channel contributes, including
/// antiphase stereo; no averaging away a voice carried by only one channel.
public struct AudioWaveform: Sendable {
  public let id = UUID()
  public let minimum: [Float]
  public let maximum: [Float]
  public let duration: Double
  /// One fixed scale per clip, not a pumping per-frame gain. Quiet speech stays
  /// legible; the 16x ceiling keeps near-silence near-flat. Audio is untouched.
  public let displayGain: Float
  public init(minimum: [Float], maximum: [Float], duration: Double) {
    self.minimum = minimum
    self.maximum = maximum
    self.duration = duration
    let negative = minimum.reduce(Float(0)) { max($0, $1.isFinite ? -$1 : 0) }
    let positive = maximum.reduce(Float(0)) { max($0, $1.isFinite ? $1 : 0) }
    let peak = max(negative, positive)
    displayGain = peak > 0 ? min(16, max(1, 0.9 / peak)) : 1
  }
}

/// Serial, off-MainActor decoding. Reads at most 16k frames at a time rather
/// than allocating a full decoded meeting. The UI caches only 16 KiB of peaks.
actor AudioWaveformDecoder {
  static let buckets = 2048
  static let chunkFrames: AVAudioFrameCount = 16384
  func decode(_ url: URL) throws -> AudioWaveform {
    try Task.checkCancellation()
    let file = try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    let format = file.processingFormat
    let length = file.length
    let duration = Double(length) / format.sampleRate
    guard length > 0, duration.isFinite, duration > 0, duration <= 3600,
      format.channelCount > 0, format.channelCount <= 8,
      Double(length) * Double(format.channelCount) <= 400_000_000,
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: Self.chunkFrames)
    else { throw WaveformUnavailable.limits }
    let count = min(Self.buckets, Int(length))
    var low = [Float](repeating: 0, count: count)
    var high = low
    var offset = 0
    var bucket = 0
    while offset < length {
      try Task.checkCancellation()
      try file.read(into: buffer, frameCount: Self.chunkFrames)
      let frames = Int(buffer.frameLength)
      guard frames > 0, let channels = buffer.floatChannelData else {
        throw WaveformUnavailable.incomplete
      }
      let end = min(Int(length), offset + frames)
      var position = offset
      while position < end {
        let bucketEnd = (bucket + 1) * Int(length) / count
        let stop = min(end, bucketEnd)
        let size = vDSP_Length(stop - position)
        for channel in 0..<Int(format.channelCount) {
          let start = channels[channel].advanced(by: position - offset)
          var minimum: Float = 0
          var maximum: Float = 0
          vDSP_minv(start, 1, &minimum, size)
          vDSP_maxv(start, 1, &maximum, size)
          if minimum.isFinite { low[bucket] = min(low[bucket], max(-1, minimum)) }
          if maximum.isFinite { high[bucket] = max(high[bucket], min(1, maximum)) }
        }
        position = stop
        if stop == bucketEnd { bucket += 1 }
      }
      offset = end
    }
    try Task.checkCancellation()
    return AudioWaveform(minimum: low, maximum: high, duration: duration)
  }
}

enum WaveformUnavailable: Error { case limits, incomplete }
