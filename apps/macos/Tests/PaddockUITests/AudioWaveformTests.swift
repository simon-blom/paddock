import AVFoundation
import Foundation
import Testing

@testable import PaddockStudio

@Suite("Bounded native audio waveforms", .serialized)
struct AudioWaveformTests {
  @Test func quietClipScalingIsBoundedVisualOnlyAndNeverPumps() {
    let quiet = AudioWaveform(minimum: [-0.05, -0.02], maximum: [0.1, 0.03], duration: 1)
    #expect(abs(quiet.displayGain - 9) < 0.0001)
    #expect(quiet.minimum == [-0.05, -0.02] && quiet.maximum == [0.1, 0.03])
    #expect(AudioWaveform(minimum: [0], maximum: [0], duration: 1).displayGain == 1)
    #expect(AudioWaveform(minimum: [-1], maximum: [1], duration: 1).displayGain == 1)
    #expect(AudioWaveform(minimum: [-0.00001], maximum: [0.00001], duration: 1).displayGain == 16)
  }
  @Test func matchesActualWebPeakReductionForMonoStereoAndSilence() async throws {
    struct Fixture: Decodable {
      let name: String
      let channels: [[Float]]
      let minimum: [Float]
      let maximum: [Float]
    }
    let resource = try #require(
      Bundle.module.url(
        forResource: "audio-waveform", withExtension: "json", subdirectory: "Fixtures"))
    for row in try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: resource)) {
      let url = try waveFile(row.channels)
      defer { try? FileManager.default.removeItem(at: url) }
      let original = try Data(contentsOf: url)
      let peaks = try await AudioWaveformDecoder().decode(url)
      #expect(peaks.minimum == row.minimum, "\(row.name)")
      #expect(peaks.maximum == row.maximum, "\(row.name)")
      #expect(peaks.duration == Double(row.minimum.count) / 16000)
      #expect(try Data(contentsOf: url) == original)
    }
  }
  @Test func exactPeaksSurviveChunkAndBucketBoundaries() async throws {
    var samples = [Float](repeating: 0, count: 50003)
    for index in [0, 23, 16383, 16384, 32767, 50002] {
      samples[index] = index.isMultiple(of: 2) ? 0.9 : -0.8
    }
    let url = try waveFile([samples])
    defer { try? FileManager.default.removeItem(at: url) }
    let wave = try await AudioWaveformDecoder().decode(url)
    #expect(wave.minimum.count == 2048 && wave.maximum.count == 2048)
    for bucket in 0..<2048 {
      let range = (bucket * samples.count / 2048)..<((bucket + 1) * samples.count / 2048)
      #expect(wave.minimum[bucket] == min(0, samples[range].min()!))
      #expect(wave.maximum[bucket] == max(0, samples[range].max()!))
    }
  }
  @Test @MainActor func previewAndPlaybackShareFetchAndHaveIndependentFileLifetimes() async throws {
    let original = try waveFile([[Float](repeating: 0.2, count: 16000)])
    defer { try? FileManager.default.removeItem(at: original) }
    let before = try Data(contentsOf: original)
    let media = StudioAudioMedia()
    var calls = 0
    let fetch: @MainActor () async throws -> URL = {
      calls += 1
      try await Task.sleep(for: .milliseconds(20))
      let copy = FileManager.default.temporaryDirectory.appendingPathComponent(
        "Paddock-wave-test-\(UUID()).wav")
      try FileManager.default.copyItem(at: original, to: copy)
      return copy
    }
    let waveTask = Task { await media.waveform(id: "one", fetch: fetch) }
    let playTask = Task { try await media.playbackCopy(id: "one", fetch: fetch) }
    let wave = await waveTask.value
    let playable = try await playTask.value
    defer { try? FileManager.default.removeItem(at: playable) }
    #expect(calls == 1 && wave?.duration == 1)
    #expect(await media.waveform(id: "one", fetch: fetch)?.id == wave?.id)
    #expect(calls == 1)
    media.reset()
    #expect(
      try Data(contentsOf: playable) == before, "Eviction cannot delete AVPlayer's leased copy")
    #expect(try Data(contentsOf: original) == before)
  }
  @Test @MainActor func cancelledPreviewDoesNotCancelAnotherReaderOrPublishAfterReset() async throws
  {
    let media = StudioAudioMedia()
    var copies: [URL] = []
    let fetch: @MainActor () async throws -> URL = {
      try await Task.sleep(for: .milliseconds(30))
      let url = try waveFile([[Float](repeating: 0.1, count: 16000)])
      copies.append(url)
      return url
    }
    let first = Task { await media.waveform(id: "one", fetch: fetch) }
    let second = Task { await media.waveform(id: "one", fetch: fetch) }
    try await Task.sleep(for: .milliseconds(5))
    first.cancel()
    #expect(await first.value == nil)
    #expect(await second.value != nil)
    let stale = Task { await media.waveform(id: "two", fetch: fetch) }
    try await Task.sleep(for: .milliseconds(5))
    media.reset()
    #expect(await stale.value == nil)
    #expect(copies.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
  }
  @Test @MainActor func rapidSeeksKeepNewestTargetAndPrepareCoalesces() async throws {
    let clip = try JSONDecoder().decode(
      StudioState.AudioClip.self,
      from: Data(#"{"id":"fixture","name":"Fixture.wav","mime":"audio/wav"}"#.utf8))
    let player = StudioAudioPlayback()
    defer { player.reset() }
    var calls = 0
    let load: @MainActor () async throws -> URL = {
      calls += 1
      try await Task.sleep(for: .milliseconds(20))
      return try waveFile([[Float](repeating: 0.1, count: 32000)])
    }
    let one = Task { await player.prepare(clip, load: load) }
    let two = Task { await player.prepare(clip, load: load) }
    #expect(await one.value)
    #expect(await two.value)
    #expect(calls == 1 && !player.playing)
    for step in 0..<100 { player.seek(Double(step) / 100) }
    #expect(player.renderingPosition == 0.99)
    try await Task.sleep(for: .milliseconds(150))
    #expect(abs(player.position - 0.99) < 0.01 && !player.playing)
    player.seek(.nan)
    #expect(abs(player.position - 0.99) < 0.01)
    player.reset()
    #expect(player.position == 0)
  }
  @Test func compressedAudioAndCancellationAreSafe() async throws {
    let wav = try waveFile([[Float](repeating: 0.25, count: 16000)])
    defer { try? FileManager.default.removeItem(at: wav) }
    let source = try AVAudioFile(forReading: wav)
    let compressed = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Paddock-wave-test-\(UUID()).m4a")
    defer { try? FileManager.default.removeItem(at: compressed) }
    do {
      let output = try AVAudioFile(
        forWriting: compressed,
        settings: [
          AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1,
          AVEncoderBitRateKey: 32000,
        ])
      let buffer = try #require(
        AVAudioPCMBuffer(pcmFormat: source.processingFormat, frameCapacity: 16000))
      try source.read(into: buffer)
      try output.write(from: buffer)
    }
    let peaks = try await AudioWaveformDecoder().decode(compressed)
    #expect(peaks.duration > 0.9 && peaks.duration < 1.2)
    let cancelled = Task {
      try Task.checkCancellation()
      return try await AudioWaveformDecoder().decode(wav)
    }
    cancelled.cancel()
    await #expect(throws: CancellationError.self) { try await cancelled.value }
  }

  @Test @MainActor func longRecordingIsBoundedAndDoesNotBlockTheMainActor() async throws {
    let url = try await Task.detached { () throws -> URL in
      let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        "Paddock-wave-test-\(UUID()).wav")
      let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16000)!
      buffer.frameLength = 16000
      for i in 0..<16000 { buffer.floatChannelData![0][i] = sin(Float(i) * 0.05) * 0.5 }
      let file = try AVAudioFile(
        forWriting: url,
        settings: [
          AVFormatIDKey: kAudioFormatLinearPCM,
          AVSampleRateKey: 16000, AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
          AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
        ])
      for _ in 0..<1800 { try file.write(from: buffer) }
      return url
    }.value
    defer { try? FileManager.default.removeItem(at: url) }
    var heartbeats = 0
    let heartbeat = Task { @MainActor in
      while !Task.isCancelled {
        heartbeats += 1
        do { try await Task.sleep(for: .milliseconds(1)) } catch { break }
      }
    }
    defer { heartbeat.cancel() }
    let start = ContinuousClock.now
    let wave = try await AudioWaveformDecoder().decode(url)
    let elapsed = start.duration(to: .now)
    #expect(wave.duration == 1800)
    #expect(wave.minimum.count == 2048 && wave.maximum.count == 2048)
    #expect(heartbeats >= 2, "The MainActor must keep scheduling during file decoding")
    print(
      "WAVEFORM_BENCH 30-minute 16-kHz mono: \(elapsed), main-actor ticks: \(heartbeats), peak payload: \((wave.minimum.count + wave.maximum.count) * MemoryLayout<Float>.stride) bytes"
    )
  }

  @Test @MainActor func cacheEvictionAndCancelledQueueKeepOnlyBoundedPrivateFiles() async throws {
    let media = StudioAudioMedia()
    var files: [URL] = []
    let fetch: @MainActor () async throws -> URL = {
      let file = try waveFile([[Float](repeating: 0.2, count: 16000)])
      files.append(file)
      return file
    }
    for index in 0..<5 { #expect(await media.waveform(id: "\(index)", fetch: fetch) != nil) }
    #expect(files.filter { FileManager.default.fileExists(atPath: $0.path) }.count == 2)
    let cached = await media.waveform(id: "0", fetch: fetch)
    #expect(cached != nil && files.count == 5, "Peaks survive source-file eviction")
    let jobs = (0..<20).map { index in
      Task {
        await media.waveform(
          id: "queued-\(index)",
          fetch: {
            try await Task.sleep(for: .milliseconds(30))
            return try await fetch()
          })
      }
    }
    try await Task.sleep(for: .milliseconds(5))
    for job in jobs { job.cancel() }
    for job in jobs { #expect(await job.value == nil) }
    #expect(files.count <= 6, "Cancelled queued previews must not all download")
    media.reset()
    #expect(files.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
  }

  private func waveFile(_ channels: [[Float]]) throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
      "Paddock-wave-test-\(UUID()).wav")
    let format = try #require(
      AVAudioFormat(
        standardFormatWithSampleRate: 16000, channels: AVAudioChannelCount(channels.count)))
    let buffer = try #require(
      AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(channels[0].count)))
    buffer.frameLength = buffer.frameCapacity
    for c in channels.indices {
      channels[c].withUnsafeBufferPointer {
        buffer.floatChannelData![c].update(from: $0.baseAddress!, count: $0.count)
      }
    }
    let file = try AVAudioFile(forWriting: url, settings: format.settings)
    try file.write(from: buffer)
    return url
  }
}
