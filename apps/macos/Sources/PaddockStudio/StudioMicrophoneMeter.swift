import Foundation
import os

/// A latest-value mailbox, not conversation state. Capture writes nine numbers
/// from its private queue; only the visible meter reads them at display cadence.
/// Slow inference/sockets cannot stall it or enqueue old animation frames.
public final class StudioMicrophoneMeter: Sendable {
  public static let bandCount = 9
  private struct Snapshot: Sendable {
    var levels = [Double](repeating: 0, count: bandCount)
    var time = ContinuousClock.now
  }
  private let latest = OSAllocatedUnfairLock(initialState: Snapshot())
  public init() {}

  func publish(_ levels: [Double], at time: ContinuousClock.Instant = .now) {
    guard levels.count == Self.bandCount else { return }
    latest.withLock { $0 = Snapshot(levels: levels, time: time) }
  }
  func reset() { latest.withLock { $0 = Snapshot() } }

  public func levels(at time: ContinuousClock.Instant = .now) -> [Double] {
    let snapshot = latest.withLock { $0 }
    let age = snapshot.time.duration(to: time).components
    let seconds = Double(age.seconds) + Double(age.attoseconds) / 1e18
    // A stopped/disconnected capture must not leave a frozen "hearing you"
    // indication. Normal callbacks get a short grace period, then the same
    // release as real silence. There is no synthetic motion in either case.
    let decay = exp(-max(0, seconds - 0.08) / MicrophoneSpectrumShape.release)
    return snapshot.levels.map { $0.isFinite ? min(1, max(0, $0 * decay)) : 0 }
  }
}

/// Matches web useMicLevels.ts: log-spaced voice bands, bounded pink-noise tilt,
/// decibel-scaled analyser bins and a time-based fast-attack/slow-release envelope.
struct MicrophoneSpectrumShape {
  static let minimumDB = -85.0
  static let maximumDB = -25.0
  static let attack = 0.025
  static let release = 0.22
  struct Band {
    let range: Range<Int>
    let gain: Double
  }
  let bands: [Band]
  private(set) var levels = [Double](repeating: 0, count: StudioMicrophoneMeter.bandCount)

  init(sampleRate: Double = 16000, binCount: Int = 512) {
    bands = (0..<StudioMicrophoneMeter.bandCount).map { band in
      let low = 80 * pow(6000 / 80, Double(band) / 9)
      let high = 80 * pow(6000 / 80, Double(band + 1) / 9)
      let start = min(binCount - 1, max(0, Int(floor(low / (sampleRate / 2) * Double(binCount)))))
      let end = min(binCount, max(start + 1, Int(ceil(high / (sampleRate / 2) * Double(binCount)))))
      return Band(range: start..<end, gain: min(2.4, pow(sqrt(low * high) / 80, 0.34)))
    }
  }
  mutating func update(bins: [UInt8], elapsed: Double) {
    let dt = min(0.1, max(0, elapsed))
    for (i, band) in bands.enumerated() {
      let sum = band.range.reduce(0.0) { $0 + Double(bins[$1]) }
      let target = min(1, sum / Double(band.range.count) / 255 * band.gain)
      let coefficient = 1 - exp(-dt / (target > levels[i] ? Self.attack : Self.release))
      levels[i] += (target - levels[i]) * coefficient
    }
  }
}
