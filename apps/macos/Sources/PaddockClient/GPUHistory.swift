import Foundation

public enum GPUHistoryMetric: String, CaseIterable, Identifiable, Sendable {
  case allocations, throughput, utilization, power, temperature
  public var id: Self { self }
  public var title: String {
    switch self {
    case .allocations: "Allocations"
    case .throughput: "Throughput"
    case .utilization: "Utilization"
    case .power: "Power"
    case .temperature: "Temperature"
    }
  }
  public var unit: String {
    switch self {
    case .allocations: "GiB"
    case .throughput: "tok/s"
    case .utilization: "%"
    case .power: "W"
    case .temperature: "°C"
    }
  }
}

public struct GPUHistoryPoint: Identifiable, Sendable {
  public let id: UInt64
  public let date: Date
  /// Changes at sleep/disconnect/clock reversal and device/runner replacement.
  public let segment: UInt64
  public let values: [GPUHistoryMetric: Double]
}

/// Bounded app-owned history. No view timers, files, credentials or WebKit.
/// Dates are sample receipt times, never synthetic evenly spaced points.
public struct GPUHistory: Sendable {
  public private(set) var points: [GPUHistoryPoint] = []
  private var serial: UInt64 = 0
  private var segment: UInt64 = 0
  private var identity = ""
  private var timestamp: UInt64?
  public init() {}

  public mutating func ingest(_ snapshot: GPUSnapshot?, at date: Date = Date()) {
    guard let snapshot, snapshot.available,
      abs(Double(snapshot.ts) - date.timeIntervalSince1970) <= 10,
      let gpu = snapshot.gpus.first
    else {
      disconnect()
      return
    }
    let runners = snapshot.reconciliation?.runners ?? []
    let key = gpu.name + runners.map(\.id).sorted().joined(separator: ",")
    let elapsed = points.last.map { date.timeIntervalSince($0.date) } ?? 0
    if elapsed < 0 { points.removeAll(keepingCapacity: true) }
    if key != identity || elapsed > 12 || elapsed < 0 || snapshot.ts < (timestamp ?? 0) {
      segment &+= 1
    } else if timestamp == snapshot.ts {
      return
    }
    identity = key
    timestamp = snapshot.ts
    var values: [GPUHistoryMetric: Double] = [:]
    if gpu.metal != nil {
      values[.allocations] = snapshot.metalAllocatedBytes.map { Double($0) / 1_073_741_824 }
    } else {
      values[.allocations] = gpu.memUsed.map { Double($0) / 1_073_741_824 }
    }
    values[.utilization] = gpu.utilGpu
    values[.power] = gpu.powerW
    values[.temperature] = gpu.tempC
    if let recon = snapshot.reconciliation,
      abs(Double(snapshot.ts) - Double(recon.ts)) <= 10,
      !runners.isEmpty, runners.allSatisfy({ $0.engine != nil })
    {
      values[.throughput] = runners.reduce(0) {
        $0 + ($1.engine?.phase == "idle" ? 0 : ($1.engine?.tokS ?? 0))
      }
    }
    values = values.filter { $0.value.isFinite && $0.value >= 0 }
    serial &+= 1
    points.append(GPUHistoryPoint(id: serial, date: date, segment: segment, values: values))
    if points.count > 900 { points.removeFirst(points.count - 900) }
  }

  public mutating func disconnect() {
    segment &+= 1
    timestamp = nil
  }
}
