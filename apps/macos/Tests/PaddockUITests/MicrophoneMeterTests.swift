import AppKit
import SwiftUI
import Testing

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native microphone signal meter", .serialized)
struct MicrophoneMeterTests {
  @Test func silenceAndDCStayQuietWithoutManufacturedBars() {
    for amplitude in [0.0, 0.01, 0.9] {
      let spectrum = NativeMicrophoneSpectrum()
      // A non-zero DC signal starts with a real step transient. Let its release
      // settle; the steady offset itself must not keep any voice band lit.
      spectrum.consume(pcm(frequency: 0, amplitude: amplitude, count: 32000, dc: true))
      #expect(spectrum.levels.allSatisfy { $0 < 0.01 })
    }
  }

  @Test func quietSpeechIsVisibleAndBandsFollowRealFrequency() {
    var strongest: [Int] = []
    for frequency in [125.0, 500.0, 2000.0, 4000.0] {
      let spectrum = NativeMicrophoneSpectrum()
      let original = pcm(frequency: frequency, amplitude: 0.015, count: 16000)
      let before = original
      spectrum.consume(original)
      let maximum = spectrum.levels.max() ?? 0
      // A pure high tone occupies fewer bins of its wider logarithmic band.
      // Unlike broadband speech it should not light that entire band strongly.
      #expect(maximum > 0.02 && maximum < 1)
      if frequency <= 500 { #expect(StudioMicrophoneLevelView.height(maximum) > 4.5) }
      #expect(spectrum.levels.min()! < maximum / 2, "Bars must not repeat one RMS value")
      #expect(original == before, "Metering cannot change saved/transmitted PCM")
      strongest.append(spectrum.levels.firstIndex(of: maximum)!)
    }
    #expect(zip(strongest, strongest.dropFirst()).allSatisfy { $0 < $1 })
  }

  @Test func louderSpeechGrowsButDoesNotAmplifyTheRecording() {
    var peaks: [Double] = []
    for amplitude in [0.003, 0.015, 0.08, 0.6] {
      let spectrum = NativeMicrophoneSpectrum()
      spectrum.consume(pcm(frequency: 250, amplitude: amplitude, count: 16000))
      peaks.append(spectrum.levels.max()!)
      #expect(spectrum.levels.allSatisfy { $0.isFinite && (0...1).contains($0) })
    }
    #expect(zip(peaks, peaks.dropFirst()).allSatisfy { $0 < $1 })
  }

  @Test func chunkBoundariesDoNotLoseSamplesOrChangeLevels() {
    let all = pcm(frequency: 875, amplitude: 0.03, count: 16037)
    let single = NativeMicrophoneSpectrum()
    let chunked = NativeMicrophoneSpectrum()
    single.consume(all)
    for start in stride(from: 0, to: all.count, by: 514) {
      chunked.consume(all.subdata(in: start..<min(all.count, start + 514)))
    }
    #expect(single.levels == chunked.levels)
    #expect(!chunked.consume(Data()))
  }

  @Test func attackIsFastAndReleaseIsSmoothThenSilent() {
    var shape = MicrophoneSpectrumShape()
    let loud = [UInt8](repeating: 200, count: 512)
    let silence = [UInt8](repeating: 0, count: 512)
    shape.update(bins: loud, elapsed: 0.05)
    let loudest = shape.levels.max()!
    #expect(loudest > 0.8)
    shape.update(bins: silence, elapsed: 0.05)
    #expect(shape.levels.max()! > loudest * 0.7)
    #expect(shape.levels.max()! < loudest)
    for _ in 0..<30 { shape.update(bins: silence, elapsed: 0.05) }
    #expect(shape.levels.max()! < 0.001)
  }

  @Test func latestCaptureReplacesBacklogAndStaleCaptureDecays() {
    let meter = StudioMicrophoneMeter()
    let now = ContinuousClock.now
    #expect(meter.levels().allSatisfy { $0 == 0 })
    meter.publish([Double](repeating: 0.2, count: 9), at: now)
    meter.publish([Double](repeating: 0.8, count: 9), at: now)
    #expect(meter.levels(at: now).allSatisfy { $0 == 0.8 })
    let later = meter.levels(at: now + .milliseconds(300))
    #expect(later.allSatisfy { abs($0 - 0.8 / exp(1)) < 1e-9 })
    #expect(meter.levels(at: now + .seconds(3)).allSatisfy { $0 < 0.00001 })
    meter.reset()
    #expect(meter.levels().allSatisfy { $0 == 0 })
  }

  @Test func sameBandEnvelopeFixturesAsWeb() throws {
    struct Fixture: Decodable {
      struct Signal: Decodable { let start: Int, end: Int, value: UInt8 }
      let name: String, sampleRate: Double, signal: [Signal], frames: Int
    }
    let url = try #require(
      Bundle.module.url(
        forResource: "microphone-meter", withExtension: "json", subdirectory: "Fixtures"))
    for fixture in try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url)) {
      var bins = [UInt8](repeating: 0, count: 512)
      for signal in fixture.signal {
        for index in signal.start..<signal.end { bins[index] = signal.value }
      }
      var shape = MicrophoneSpectrumShape(sampleRate: fixture.sampleRate)
      for frame in 0..<fixture.frames {
        shape.update(bins: bins, elapsed: frame == 0 ? 1 / 60 : 0.016)
      }
      // Independent analytic envelope for a steady spectrum. Web consumes the
      // same fixtures through its actual analyser/sampling path.
      let time = 1.0 / 60 + Double(fixture.frames - 1) * 0.016
      for (i, band) in shape.bands.enumerated() {
        let mean = band.range.reduce(0.0) { $0 + Double(bins[$1]) } / Double(band.range.count) / 255
        let expected = min(1, mean * band.gain) * (1 - exp(-time / 0.025))
        #expect(abs(shape.levels[i] - expected) < 1e-9, "\(fixture.name), band \(i)")
      }
    }
  }

  @Test @MainActor func meterFitsBothAppearancesWithoutVisibleTestWindows() {
    for dark in [false, true] {
      let meter = StudioMicrophoneMeter()
      meter.publish([0, 0.15, 0.3, 0.6, 0.85, 0.5, 0.3, 0.1, 0])
      let host = NSHostingView(
        rootView: StudioMicrophoneLevelView(meter: meter)
          .environment(\.colorScheme, dark ? .dark : .light))
      #expect(host.fittingSize == NSSize(width: 43, height: 18))
    }
    #expect(StudioMicrophoneLevelView.height(0) == 2.88)
    #expect(StudioMicrophoneLevelView.height(1) == 18)
    #expect(StudioMicrophoneLevelView.height(.nan) == 2.88)
  }

  @Test func captureSideAnalysisBudget() {
    let spectrum = NativeMicrophoneSpectrum()
    let frame = pcm(frequency: 500, amplitude: 0.03, count: 256)
    var milliseconds: [Double] = []
    for _ in 0..<625 {
      let start = ContinuousClock.now
      spectrum.consume(frame)
      let elapsed = start.duration(to: .now).components
      milliseconds.append(Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15)
    }
    // Generous regression guard, far below a 16 ms capture hop, not a rival bar.
    #expect(milliseconds.sorted()[618] < 8)
    if ProcessInfo.processInfo.environment["PADDOCK_METER_DIAGNOSTICS"] != nil {
      print(
        "Native meter 625 hops: p50=\(milliseconds.sorted()[312]) ms, p99=\(milliseconds.sorted()[618]) ms"
      )
    }
  }

  private func pcm(frequency: Double, amplitude: Double, count: Int, dc: Bool = false) -> Data {
    var samples = (0..<count).map { index in
      Int16(
        max(
          -32768,
          min(
            32767,
            (amplitude * (dc ? 1 : sin(2 * .pi * frequency * Double(index) / 16000)) * 32768)
              .rounded()))
      ).littleEndian
    }
    return samples.withUnsafeMutableBytes { Data($0) }
  }
}
