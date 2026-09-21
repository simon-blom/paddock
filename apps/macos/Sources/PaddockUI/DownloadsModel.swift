import Foundation
import Observation
import PaddockClient

/// One retained app-owned projection. No transfers or per-byte SQLite writes
/// happen on MainActor. A single light poll exists only while work is active.
@MainActor @Observable
public final class DownloadsModel {
  public private(set) var jobs: [ModelDownload] = []
  public private(set) var error: String?
  public private(set) var busy = false
  public private(set) var loaded = false
  private var activity: [String: DownloadActivity] = [:]
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private let now: () -> TimeInterval
  @ObservationIgnored private var monitor: Task<Void, Never>?
  @ObservationIgnored private var stopped = false
  @ObservationIgnored private var generation: UInt64 = 0
  @ObservationIgnored var onCompletion: (() async -> Void)?

  public init(
    client: any ManagerLoading,
    now: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
  ) {
    self.client = client
    self.now = now
  }
  public var active: Bool { jobs.contains(where: \.active) }
  func transferStatus(for job: ModelDownload) -> String? {
    guard job.active, job.cancelling != true, job.stage != "verifying",
      job.downloaded < job.total, let sample = activity[job.id]
    else { return nil }
    if sample.waiting { return "Waiting for download data…" }
    guard sample.bytesPerSecond > 0 else { return "Measuring download speed…" }
    return "\(DisplayFormat.bytes(UInt64(min(sample.bytesPerSecond, Double(Int64.max))))) / s"
  }
  var visible: [ModelDownload] {
    // Resume returns a new job ID in the shared protocol. Show the latest
    // attempt per exact selection rather than stale Pause rows for old attempts.
    var seen = Set<String>()
    return jobs.sorted { $0.createdMs > $1.createdMs }.filter {
      seen.insert($0.model + ":" + ($0.artifacts ?? []).sorted().joined(separator: ",")).inserted
    }
  }

  func plan(model: String, artifact: String) async throws -> DownloadPlan {
    let reply = try await client.downloads(.plan(model: model, artifact: artifact))
    guard let plan = reply.plan else {
      throw ManagerError.core("Download details are unavailable.")
    }
    return plan
  }

  @discardableResult
  func perform(_ command: DownloadCommand) async -> Bool {
    guard !busy, !stopped else { return false }
    busy = true
    generation &+= 1
    let attempt = generation
    error = nil
    defer { busy = false }
    do {
      let reply = try await client.downloads(command)
      guard !stopped, attempt == generation else { return false }
      await apply(reply.jobs)
      return true
    } catch {
      guard !stopped, attempt == generation else { return false }
      self.error = error.localizedDescription
      return false
    }
  }

  func refresh() async {
    guard !busy, !stopped else { return }
    generation &+= 1
    let attempt = generation
    do {
      let reply = try await client.downloads(.list)
      guard !stopped, attempt == generation else { return }
      error = nil
      await apply(reply.jobs)
    } catch {
      guard !stopped, attempt == generation else { return }
      self.error = error.localizedDescription
    }
  }

  private func apply(_ next: [ModelDownload]) async {
    let completed =
      loaded
      && next.contains { job in
        job.complete && !jobs.contains { $0.id == job.id && $0.complete }
      }
    let time = now()
    let transferring = next.filter {
      $0.active && $0.stage != "verifying" && $0.downloaded < $0.total && $0.cancelling != true
    }
    activity = activity.filter { id, _ in transferring.contains { $0.id == id } }
    for job in transferring {
      activity[job.id, default: DownloadActivity()].record(bytes: job.downloaded, at: time)
    }
    jobs = next
    loaded = true
    if active && monitor == nil {
      monitor = Task { [weak self] in
        while !Task.isCancelled {
          do { try await Task.sleep(for: .seconds(1)) } catch { break }
          guard let self, !self.stopped else { break }
          await self.refresh()
          // On a failed probe keep retrying; do not turn stale progress into Done.
          if !self.active && self.error == nil { break }
        }
        self?.monitor = nil
      }
    }
    if completed { await onCompletion?() }
  }

  func stop() {
    stopped = true
    generation &+= 1
    monitor?.cancel()
    monitor = nil
  }
}

/// A bounded five-second throughput window. Recovered bytes aren't network
/// throughput; retries may roll live bytes back without changing resume bits.
struct DownloadActivity {
  private var samples: [(time: TimeInterval, bytes: UInt64)] = []
  private var lastChange: TimeInterval = 0
  private(set) var bytesPerSecond: Double = 0
  private(set) var waiting = false

  mutating func record(bytes: UInt64, at time: TimeInterval) {
    guard let previous = samples.last, bytes >= previous.bytes, time > previous.time else {
      samples = [(time, bytes)]
      lastChange = time
      bytesPerSecond = 0
      waiting = false
      return
    }
    if bytes > previous.bytes { lastChange = time }
    samples.append((time, bytes))
    while samples.count > 2 && (samples[1].time <= time - 5 || samples.count > 16) {
      samples.removeFirst()
    }
    let first = samples[0]
    waiting = time - lastChange >= 5
    bytesPerSecond = waiting ? 0 : Double(bytes - first.bytes) / (time - first.time)
  }
}
