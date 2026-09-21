import Foundation
import Observation
import PaddockClient

struct EndpointLogLine: Identifiable, Sendable, Equatable {
  let id: UInt64
  let raw: String
  let level: String?
  let effective: String?
  let clock: String?
  let module: String?
  let message: String?
  var display: String {
    guard let clock, let level else { return raw + "\n" }
    return
      "\(clock)  \(level.padding(toLength: 5, withPad: " ", startingAt: 0))  \(message ?? raw)\n"
  }
}

/// Mirrors LogView.vue: ANSI removal, tracing timestamps/levels, inherited
/// continuation severity, panic promotion and newest-matching limits.
struct EndpointLogBuffer: Sendable {
  private(set) var lines: [EndpointLogLine] = []
  private(set) var bytes = 0
  private var nextID: UInt64 = 0
  private var previousLevel: String?
  private static let ansi = try! NSRegularExpression(
    pattern: #"\x{1b}(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x{7}\x{1b}]*(?:\x{7}|\x{1b}\\)|[@-Z\\-_])"#)
  private static let controls = try! NSRegularExpression(
    pattern: #"[\x{0}-\x{8}\x{b}-\x{1f}\x{7f}]"#)
  private static let tracing = try! NSRegularExpression(
    pattern:
      #"^(?:\[([^\]]+)\]\s+)?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z?)\s+(TRACE|DEBUG|INFO|WARN|ERROR)\s+([\w:.-]+):\s?(.*)$"#
  )
  private static let panic = try! NSRegularExpression(
    pattern: #"panicked at|RUST_BACKTRACE|stack backtrace|^thread '"#, options: .caseInsensitive)
  static let ranks = ["TRACE": 0, "DEBUG": 1, "INFO": 2, "WARN": 3, "ERROR": 4]
  mutating func reset() {
    lines = []
    bytes = 0
    previousLevel = nil
  }

  mutating func append(_ text: String) {
    let iso = ISO8601DateFormatter()
    iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    let plain = ISO8601DateFormatter()
    let clock = DateFormatter()
    clock.locale = Locale(identifier: "en_US_POSIX")
    clock.dateFormat = "HH:mm:ss"
    for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
      var raw = String(rawLine)
      for regex in [Self.ansi, Self.controls] {
        raw = regex.stringByReplacingMatches(
          in: raw, range: NSRange(raw.startIndex..., in: raw), withTemplate: "")
      }
      guard !raw.isEmpty else { continue }
      var level: String?
      var time: String?
      var module: String?
      var message: String?
      if let match = Self.tracing.firstMatch(in: raw, range: NSRange(raw.startIndex..., in: raw)) {
        func group(_ index: Int) -> String {
          Range(match.range(at: index), in: raw).map { String(raw[$0]) } ?? ""
        }
        let timestamp = group(2)
        let stamp = timestamp.hasSuffix("Z") ? timestamp : timestamp + "Z"
        time =
          (iso.date(from: stamp) ?? plain.date(from: stamp)).map { clock.string(from: $0) }
          ?? String(timestamp.dropFirst(11).prefix(8))
        level = group(3)
        module = group(4)
        message = group(5)
      } else if Self.panic.firstMatch(in: raw, range: NSRange(raw.startIndex..., in: raw)) != nil {
        level = "ERROR"
      }
      if let level { previousLevel = level }
      nextID += 1
      lines.append(
        EndpointLogLine(
          id: nextID, raw: raw, level: level, effective: previousLevel, clock: time, module: module,
          message: message))
      bytes += raw.utf8.count
    }
    var drop = 0
    while lines.count - drop > 4000 || bytes > 1024 * 1024 {
      bytes -= lines[drop].raw.utf8.count
      drop += 1
    }
    if drop > 0 { lines.removeFirst(drop) }
  }
  func visible(query: String, minimum: String) -> [EndpointLogLine] {
    let rank = Self.ranks[minimum] ?? 0
    let query = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    return Array(
      lines.lazy.filter { line in
        (rank == 0 || (line.effective.flatMap { Self.ranks[$0] } ?? -1) >= rank)
          && (query.isEmpty || line.raw.lowercased().contains(query))
      }.suffix(1500))
  }
}

@MainActor @Observable
final class EndpointLogsModel {
  private(set) var buffer = EndpointLogBuffer()
  private(set) var state = "paused"
  private(set) var error: String?
  var query = ""
  var minimum = "all"
  var following = true
  var jump = 0
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var generation: UInt64 = 0
  init(client: any ManagerLoading) { self.client = client }
  var visible: [EndpointLogLine] { buffer.visible(query: query, minimum: minimum) }
  func latest() {
    following = true
    jump += 1
  }

  /// Owned by the visible view's task. Cancellation always closes its own
  /// receipt; late batches cannot paint a newly selected endpoint.
  func follow(port: UInt16) async {
    generation += 1
    let ticket = generation
    while !Task.isCancelled && generation == ticket {
      var id: String?
      do {
        state = "connecting"
        error = nil
        let opened = try await client.logs(.open(port: port))
        id = opened.id
        guard let subscription = id else {
          throw ManagerError.core("No log subscription returned.")
        }
        try Task.checkCancellation()
        guard generation == ticket else { throw CancellationError() }
        buffer.reset()
        while !Task.isCancelled && generation == ticket {
          let reply = try await client.logs(.poll(id: subscription))
          try Task.checkCancellation()
          guard generation == ticket else { throw CancellationError() }
          guard reply.id == subscription else {
            throw ManagerError.core("Unexpected log subscription.")
          }
          state = reply.state
          if reply.state == "failed" { throw ManagerError.core("The log reader stopped.") }
          if !reply.text.isEmpty {
            let previous = buffer
            let updated = await Task.detached(priority: .utility) {
              var value = previous
              value.append(reply.text)
              return value
            }.value
            try Task.checkCancellation()
            guard generation == ticket else { throw CancellationError() }
            buffer = updated
          }
          try await Task.sleep(for: .milliseconds(300))
        }
      } catch {
        if !Task.isCancelled && generation == ticket {
          self.error = "Log connection interrupted. Reconnecting…"
          state = "connecting"
        }
      }
      if let id { _ = try? await client.logs(.close(id: id)) }
      if Task.isCancelled || generation != ticket { break }
      try? await Task.sleep(for: .seconds(2))
    }
    if generation == ticket { state = "paused" }
  }
}
