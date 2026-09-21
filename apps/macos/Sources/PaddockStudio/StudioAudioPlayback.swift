import AVFoundation
import Foundation
import Observation
import PaddockClient

/// One transport for the user clip and every comparison lane. Audio bytes never
/// cross the JavaScript bridge. Only one downloaded original is retained, and
/// its lifetime is independent of the permanent attachment in the Rust store.
@MainActor @Observable public final class StudioAudioPlayback {
  public private(set) var clipId: String?
  public private(set) var position = 0.0
  public private(set) var duration = 0.0
  public private(set) var playing = false
  public private(set) var loading = false
  public private(set) var error: String?
  public var rate: Float = 1 { didSet { if playing { player.rate = rate } } }
  public var volume: Float = 1 { didSet { player.volume = volume } }
  public var muted = false { didSet { player.isMuted = muted } }
  @ObservationIgnored private let player = AVPlayer()
  @ObservationIgnored private var observer: Any?
  @ObservationIgnored private var file: URL?
  @ObservationIgnored private var pending: Task<Bool, Never>?
  @ObservationIgnored private var seekTarget: Double?
  @ObservationIgnored private var seekInFlight = false
  @ObservationIgnored private var ended: NSObjectProtocol?
  private var epoch = 0
  public init() {}

  /// Sample AVPlayer's actual clock in the waveform/transcript display islands.
  /// Publishing every frame through the shared player would invalidate the
  /// whole conversation for each playhead or karaoke-word update.
  public var renderingPosition: Double {
    if let seekTarget { return seekTarget }
    guard playing else { return position }
    let seconds = player.currentTime().seconds
    return seconds.isFinite ? max(0, seconds) : position
  }

  @discardableResult public func prepare(
    _ clip: StudioState.AudioClip, load: @escaping @MainActor () async throws -> URL
  ) async -> Bool {
    if clipId == clip.id, player.currentItem != nil { return true }
    if clipId == clip.id, let pending { return await pending.value }
    reset()
    clipId = clip.id
    loading = true
    let ticket = epoch
    let task = Task { [weak self] in
      guard let self else { return false }
      return await loadItem(clip, ticket: ticket, load: load)
    }
    pending = task
    return await task.value
  }
  private func loadItem(
    _ clip: StudioState.AudioClip, ticket: Int,
    load: @escaping @MainActor () async throws -> URL
  ) async -> Bool {
    do {
      let url = try await load()
      guard ticket == epoch else {
        try? FileManager.default.removeItem(at: url)
        return false
      }
      file = url
      let asset = AVURLAsset(url: url)
      let playable = try await asset.load(.isPlayable)
      let seconds = try await asset.load(.duration).seconds
      guard ticket == epoch else { return false }
      guard playable, seconds.isFinite, seconds > 0 else {
        throw ManagerError.core(
          "This recording is incomplete or uses an unsupported audio format. Its saved original has not been changed."
        )
      }
      duration = seconds
      let item = AVPlayerItem(asset: asset)
      player.replaceCurrentItem(with: item)
      player.volume = volume
      player.isMuted = muted
      ended = NotificationCenter.default.addObserver(
        forName: .AVPlayerItemDidPlayToEndTime,
        object: item, queue: .main
      ) { [weak self] _ in
        Task { @MainActor [weak self] in
          guard let self, self.epoch == ticket, self.seekTarget == nil else { return }
          self.position = self.duration
          self.playing = false
        }
      }
      observer = player.addPeriodicTimeObserver(
        forInterval: CMTime(seconds: 0.1, preferredTimescale: 600), queue: .main
      ) { [weak self] time in
        Task { @MainActor [weak self] in
          guard let self, self.epoch == ticket else { return }
          if self.seekTarget == nil {
            self.position = max(0, time.seconds.isFinite ? time.seconds : 0)
          }
          if let failure = self.player.currentItem?.error {
            self.error =
              "Audio playback failed: \(failure.localizedDescription). The saved original is unchanged."
            self.pause()
          }
        }
      }
      loading = false
      pending = nil
      return true
    } catch {
      guard ticket == epoch else { return false }
      loading = false
      pending = nil
      self.error =
        (error as? ManagerError)?.localizedDescription
        ?? "This recording could not be decoded. Its saved original has not been changed."
      if let file { try? FileManager.default.removeItem(at: file) }
      file = nil
      return false
    }
  }
  public func toggle() {
    guard player.currentItem != nil else { return }
    if playing {
      pause()
    } else {
      if position >= duration - 0.02 { seek(0) }
      playing = true
      player.playImmediately(atRate: rate)
    }
  }
  public func seek(_ seconds: Double) {
    guard seconds.isFinite, player.currentItem != nil else { return }
    position = min(duration, max(0, seconds))
    seekTarget = position
    finishLatestSeek()
  }
  private func finishLatestSeek() {
    guard !seekInFlight, let target = seekTarget else { return }
    seekInFlight = true
    let ticket = epoch
    player.seek(
      to: CMTime(seconds: target, preferredTimescale: 600),
      toleranceBefore: .zero, toleranceAfter: .zero
    ) { [weak self] completed in
      Task { @MainActor [weak self] in
        guard let self, self.epoch == ticket else { return }
        self.seekInFlight = false
        if self.seekTarget != target {
          self.finishLatestSeek()
        } else {
          self.seekTarget = nil
          let actual = self.player.currentTime().seconds
          if !completed, actual.isFinite { self.position = max(0, actual) }
        }
      }
    }
  }
  public func pause() {
    player.pause()
    playing = false
  }
  public func reset() {
    epoch += 1
    pending?.cancel()
    pending = nil
    pause()
    if let ended { NotificationCenter.default.removeObserver(ended) }
    ended = nil
    if let observer { player.removeTimeObserver(observer) }
    observer = nil
    player.replaceCurrentItem(with: nil)
    seekTarget = nil
    seekInFlight = false
    if let file { try? FileManager.default.removeItem(at: file) }
    file = nil
    clipId = nil
    position = 0
    duration = 0
    loading = false
    error = nil
  }
}
