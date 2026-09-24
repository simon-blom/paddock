import Foundation

public enum MaintenanceCommand: Encodable, Sendable {
  case usage(from: Int64, to: Int64, port: UInt16?)
  case activity(port: UInt16?, before: Int64? = nil)
  case cache
  case storage
  case benchmark(port: UInt16, pid: UInt32, concurrency: Int, long: Bool)
  case benchmarkHistory
  case exportBenchmark(id: String, path: String)
  case profiles(model: String, artifact: String)
  case saveProfile(port: UInt16, revision: String, name: String)
  case removeProfile(id: String)
  case clientInfo(port: UInt16, pid: UInt32)
  case exportCredentials(port: UInt16, pid: UInt32, path: String)
  case backup(path: String)
  case poll(String)
  case close(String)
  private enum CodingKeys: String, CodingKey {
    case kind, from, to, port, before, id, pid, path, model, artifact, name, revision, concurrency,
      long
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .usage(let from, let to, let port):
      try c.encode("usage", forKey: .kind)
      try c.encode(from, forKey: .from)
      try c.encode(to, forKey: .to)
      try c.encodeIfPresent(port, forKey: .port)
    case .activity(let port, let before):
      try c.encode("activity", forKey: .kind)
      try c.encodeIfPresent(port, forKey: .port)
      try c.encodeIfPresent(before, forKey: .before)
    case .cache: try c.encode("cache", forKey: .kind)
    case .storage: try c.encode("storage", forKey: .kind)
    case .benchmark(let port, let pid, let concurrency, let long):
      try c.encode("benchmark", forKey: .kind)
      try c.encode(port, forKey: .port)
      try c.encode(pid, forKey: .pid)
      try c.encode(concurrency, forKey: .concurrency)
      try c.encode(long, forKey: .long)
    case .benchmarkHistory: try c.encode("benchmark_history", forKey: .kind)
    case .exportBenchmark(let id, let path):
      try c.encode("export_benchmark", forKey: .kind)
      try c.encode(id, forKey: .id)
      try c.encode(path, forKey: .path)
    case .profiles(let model, let artifact):
      try c.encode("profiles", forKey: .kind)
      try c.encode(model, forKey: .model)
      try c.encode(artifact, forKey: .artifact)
    case .saveProfile(let port, let revision, let name):
      try c.encode("save_profile", forKey: .kind)
      try c.encode(port, forKey: .port)
      try c.encode(revision, forKey: .revision)
      try c.encode(name, forKey: .name)
    case .removeProfile(let id):
      try c.encode("remove_profile", forKey: .kind)
      try c.encode(id, forKey: .id)
    case .clientInfo(let port, let pid):
      try c.encode("client_info", forKey: .kind)
      try c.encode(port, forKey: .port)
      try c.encode(pid, forKey: .pid)
    case .exportCredentials(let port, let pid, let path):
      try c.encode("export_credentials", forKey: .kind)
      try c.encode(port, forKey: .port)
      try c.encode(pid, forKey: .pid)
      try c.encode(path, forKey: .path)
    case .backup(let path):
      try c.encode("backup", forKey: .kind)
      try c.encode(path, forKey: .path)
    case .poll(let id), .close(let id):
      try c.encode(id, forKey: .id)
      if case .poll = self {
        try c.encode("poll", forKey: .kind)
      } else {
        try c.encode("close", forKey: .kind)
      }
    }
  }
}

public struct MaintenanceReply: Decodable, Sendable {
  public let id: String?
  public let state: String
  public let payload: String?
  public let message: String?
}

extension ManagerLoading {
  /// Every accepted job is closed, including cancellation and decode failure.
  /// Neither sleeping nor runner I/O occupies the serial native management queue.
  public func inspect<T: Decodable & Sendable>(
    _ command: MaintenanceCommand, as type: T.Type, timeout: Duration = .seconds(30)
  ) async throws -> T {
    try Task.checkCancellation()
    var reply = try await maintenance(command)
    guard let id = reply.id else { throw ManagerError.core("Missing management receipt.") }
    do {
      let deadline = ContinuousClock.now + timeout
      while ContinuousClock.now < deadline {
        try Task.checkCancellation()
        if reply.state != "running" { break }
        try await Task.sleep(for: .milliseconds(125))
        reply = try await maintenance(.poll(id))
      }
      guard reply.state == "complete", let payload = reply.payload else {
        throw ManagerError.core(reply.message ?? "Management request timed out.")
      }
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      let value = try decoder.decode(type, from: Data(payload.utf8))
      _ = try? await maintenance(.close(id))
      return value
    } catch {
      _ = try? await maintenance(.close(id))
      throw error
    }
  }
}

public struct LocalClientSetup: Decodable, Sendable {
  public let baseUrl: String
  public let model: String
  public let hasKey: Bool
}
public struct NativeExportReceipt: Decodable, Sendable {
  public let bytes: UInt64
}

public struct UsageHistorySnapshot: Decodable, Sendable {
  public let grainMs: Int64
  public let nowMs: Int64
  public let buckets: [UsageBucket]
  public let gaps: [UsageGap]
  public let generations: [UsageGeneration]
  public let web: [UsageWebBucket]
}
public struct UsageBucket: Decodable, Sendable, Identifiable {
  public let t: Int64
  public let port: UInt16
  public let requests, errors4xx, errors5xx, disconnects: Int64
  public let inputTokens, outputTokens, cachedTokens: Int64
  public let durationMsSum: Double
  public let specDrafted, specAccepted: Int64
  // Foundation's snake-case transform capitalizes the alphabetic part after
  // a digit (errors_4xx -> errors4Xx). Keep the public property names readable.
  private enum CodingKeys: String, CodingKey {
    case t, port, requests, disconnects, inputTokens, outputTokens, cachedTokens,
      durationMsSum, specDrafted, specAccepted
    case errors4xx = "errors4Xx"
    case errors5xx = "errors5Xx"
  }
  public var id: String { "\(port):\(t)" }
  public var date: Date { Date(timeIntervalSince1970: Double(t) / 1000) }
}
public struct UsageGap: Decodable, Sendable, Identifiable {
  public let id: Int64
  public let port: UInt16
  public let fromTsMs, toTsMs: Int64
  public let cause: String
}
public struct UsageGeneration: Decodable, Sendable, Identifiable {
  public let instanceId: String
  public let port: UInt16
  public let model, embedder, asr, aligner: String?
  public let startedMs: Int64
  public let endedMs, lastSeenMs: Int64?
  public var id: String { instanceId }
  public var title: String { model ?? embedder ?? asr ?? aligner ?? "Runner" }
}
public struct UsageWebBucket: Decodable, Sendable {
  public let provider: String
  public let requests, credits, microdollars: Int64
}

public struct ActivitySnapshot: Decodable, Sendable {
  public let events: [ActivityEvent]
}
public struct ActivityEvent: Decodable, Sendable, Identifiable {
  public let fields: [String: ManagementValue]
  public init(from decoder: any Decoder) throws {
    fields = try decoder.singleValueContainer().decode([String: ManagementValue].self)
  }
  public var id: String { "\(number("port") ?? 0):\(number("ts_ms") ?? 0):\(number("seq") ?? 0)" }
  public var model: String {
    text("gen_ai.response.model") ?? text("gen_ai.request.model") ?? "Request"
  }
  public func text(_ key: String) -> String? { fields[key]?.text }
  public func number(_ key: String) -> Double? { fields[key]?.number }
}

public indirect enum ManagementValue: Codable, Sendable, Equatable {
  case string(String)
  case number(Double)
  case bool(Bool)
  case null
  case array([ManagementValue])
  case object([String: ManagementValue])
  public init(from decoder: any Decoder) throws {
    let c = try decoder.singleValueContainer()
    if c.decodeNil() {
      self = .null
    } else if let value = try? c.decode(Bool.self) {
      self = .bool(value)
    } else if let value = try? c.decode(Double.self) {
      self = .number(value)
    } else if let value = try? c.decode(String.self) {
      self = .string(value)
    } else if let value = try? c.decode([Self].self) {
      self = .array(value)
    } else {
      self = .object(try c.decode([String: Self].self))
    }
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.singleValueContainer()
    switch self {
    case .string(let v): try c.encode(v)
    case .number(let v): try c.encode(v)
    case .bool(let v): try c.encode(v)
    case .null: try c.encodeNil()
    case .array(let v): try c.encode(v)
    case .object(let v): try c.encode(v)
    }
  }
  public var text: String? { if case .string(let s) = self { s } else { nil } }
  public var number: Double? { if case .number(let n) = self, n.isFinite { n } else { nil } }
  public var display: String {
    switch self {
    case .string(let s): s
    case .number(let n): n.formatted(.number.precision(.fractionLength(0...2)))
    case .bool(let v): v ? "Yes" : "No"
    case .null: "—"
    case .array, .object:
      String(data: (try? JSONEncoder().encode(self)) ?? Data(), encoding: .utf8) ?? "—"
    }
  }
}

public struct CacheSnapshot: Decodable, Sendable {
  public let servers: [CacheInstance]
}
public struct CacheInstance: Decodable, Sendable, Identifiable {
  public let port: UInt16
  public let model: String?
  public let tier: [String: ManagementValue]
  public var id: UInt16 { port }
  public func number(_ field: String) -> Double? { tier[field]?.number }
  public var hitRate: Double? {
    guard let lookups = number("lookups"), lookups > 0, let hits = number("hits") else {
      return nil
    }
    return hits / lookups
  }
}
