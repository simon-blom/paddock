import Foundation

/// The same Rust telemetry projection used by the web Studio. Missing sensors
/// remain nil; unified RAM and runner allocations never masquerade as VRAM.
public struct GPUSnapshot: Decodable, Sendable {
  public let available: Bool
  public let ts: UInt64
  public let gpus: [GPUDeviceMetrics]
  public let reconciliation: GPUReconciliation?

  public var metalAllocatedBytes: UInt64? {
    guard let reconciliation, abs(Double(ts) - Double(reconciliation.ts)) <= 10,
      reconciliation.runners.allSatisfy({ $0.metal != nil })
    else { return nil }
    var total: UInt64 = 0
    for runner in reconciliation.runners {
      let (sum, overflow) = total.addingReportingOverflow(runner.metal?.allocatedBytes ?? 0)
      guard !overflow else { return nil }
      total = sum
    }
    return total
  }
}

public struct GPUDeviceMetrics: Decodable, Sendable, Identifiable {
  public let index: UInt32
  public let name: String
  public let metal: MetalHardwareMetrics?
  public let utilGpu: Double?
  public let memUsed: UInt64?
  public let memTotal: UInt64?
  public let powerW: Double?
  public let tempC: Double?
  public var id: UInt32 { index }
}

public struct MetalHardwareMetrics: Decodable, Sendable {
  public let unifiedMemoryTotal: UInt64?
  public let recommendedWorkingSet: UInt64
  public let thermalPressure: String?
  public let memoryPressure: String?
  public let counterSets: [String]?
}

public struct GPUReconciliation: Decodable, Sendable {
  public let ts: UInt64
  public let runners: [GPURunnerMetrics]
}

public struct GPURunnerMetrics: Decodable, Sendable, Identifiable {
  public let port: UInt16
  public let pid: UInt32
  public let selfMem: UInt64?
  public let metal: MetalRunnerMetrics?
  public let engine: GPUEngineMetrics?
  public var id: String { "\(port):\(pid)" }
}

public struct MetalRunnerMetrics: Decodable, Sendable {
  public let allocatedBytes: UInt64
  public let completedCommands: UInt64
  public let gpuSecondsTotal: Double
  public let lastCommandMs: Double?
}

public struct GPUEngineMetrics: Decodable, Sendable {
  public let tokS: Double
  public let phase: String
  public let activeSlots: UInt32
  public let kvUsed: UInt32
  public let kvTotal: UInt32
  public let tokensTotal: UInt64
}
