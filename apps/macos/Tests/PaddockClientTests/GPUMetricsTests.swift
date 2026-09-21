import Foundation
import Testing

@testable import PaddockClient

@Suite("Metal telemetry")
struct GPUMetricsTests {
  private func sample(runners: String, timestamp: UInt64 = 100) throws -> GPUSnapshot {
    try ManagerWire.decode(
      GPUSnapshot.self,
      from: Data(
        """
        {"available":true,"ts":\(timestamp),"gpus":[{"index":0,"name":"Apple M5 Max",
        "metal":{"unified_memory_total":128,"recommended_working_set":96}}],
        "reconciliation":{"ts":100,"runners":\(runners)}}
        """.utf8))
  }

  @Test func capacityIsNotUsageAndMissingSensorsStayAbsent() throws {
    let value = try sample(runners: "[]")
    #expect(value.gpus.first?.metal?.recommendedWorkingSet == 96)
    #expect(value.gpus.first?.utilGpu == nil)
    #expect(value.gpus.first?.powerW == nil)
    #expect(value.gpus.first?.memTotal == nil)
    #expect(value.metalAllocatedBytes == 0)
  }

  @Test func processTimingsSurviveTheWire() throws {
    let value = try sample(
      runners:
        #"[{"port":11540,"pid":42,"metal":{"allocated_bytes":24,"completed_commands":8,"gpu_seconds_total":0.3,"last_command_ms":20}}]"#
    )
    #expect(value.metalAllocatedBytes == 24)
    #expect(value.reconciliation?.runners.first?.metal?.lastCommandMs == 20)
  }

  @Test func oldRunnerAndStaleSamplesAreGaps() throws {
    #expect(try sample(runners: #"[{"port":11540,"pid":42}]"#).metalAllocatedBytes == nil)
    #expect(try sample(runners: "[]", timestamp: 111).metalAllocatedBytes == nil)
    #expect(try sample(runners: "[]", timestamp: 89).metalAllocatedBytes == nil)
  }

  @Test func historyBoundsDeduplicationSleepRestartAndMissingSensors() throws {
    var history = GPUHistory()
    func ingest(_ time: UInt64, _ pid: UInt32 = 42) throws {
      let snapshot = try sample(runners: "[{\"port\":11540,\"pid\":\(pid)}]", timestamp: time)
      history.ingest(snapshot, at: Date(timeIntervalSince1970: Double(time)))
    }
    try ingest(100)
    try ingest(100)
    #expect(history.points.count == 1)
    #expect(history.points[0].values[.allocations] == nil)
    #expect(history.points[0].values[.utilization] == nil)
    try ingest(102, 43)
    #expect(history.points[0].segment != history.points[1].segment)
    try ingest(200, 43)
    #expect(history.points[1].segment != history.points[2].segment)
    history.disconnect()
    try ingest(201, 43)
    #expect(history.points[2].segment != history.points[3].segment)
    // A backward clock resets the time axis, never joining future samples.
    let priorSegment = history.points.last?.segment
    try ingest(199, 43)
    #expect(history.points.count == 1)
    #expect(history.points.last?.segment != priorSegment)
    for time in 202..<1300 { try ingest(UInt64(time), 43) }
    #expect(history.points.count == 900)
    let count = history.points.count
    history.ingest(try sample(runners: "[]"), at: Date(timeIntervalSince1970: 9999))
    #expect(history.points.count == count)
  }
}
