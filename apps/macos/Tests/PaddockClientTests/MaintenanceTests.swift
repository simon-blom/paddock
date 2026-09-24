import Foundation
import Testing

@testable import PaddockClient

@Suite("Native management jobs")
struct MaintenanceTests {
  @Test func usesBoundedTypedCommandsAndClosesSuccessfulJobs() async throws {
    let client = MaintenanceFixture(payload: "{\"bytes\":123}")
    let result = try await client.inspect(
      .backup(path: "/tmp/chosen.sqlite"), as: NativeExportReceipt.self)
    #expect(result.bytes == 123)
    #expect(await client.closed)
    let value =
      try JSONSerialization.jsonObject(
        with: JSONEncoder().encode(MaintenanceCommand.usage(from: 1, to: 2000, port: 11540)))
      as? [String: Any]
    #expect(value?["kind"] as? String == "usage")
    #expect(value?["port"] as? Int == 11540)
    #expect(value?["url"] == nil)
  }
  @Test func decodeFailureAndCancellationCloseTheirJob() async {
    let invalid = MaintenanceFixture(payload: "{bad")
    do {
      _ = try await invalid.inspect(.cache, as: CacheSnapshot.self)
      Issue.record("Invalid JSON accepted")
    } catch {}
    #expect(await invalid.closed)
    let pending = MaintenanceFixture(payload: nil)
    let task = Task { try await pending.inspect(.cache, as: CacheSnapshot.self) }
    while !(await pending.started) { await Task.yield() }
    task.cancel()
    _ = try? await task.value
    #expect(await pending.closed)
  }
  @Test func rawActivityAndCacheNamesDoNotLoseUnderscoresOrMissingMeasurements() throws {
    let data = Data(
      #"{"events":[{"port":11540,"seq":2,"ts_ms":123,"gen_ai.request.model":"fixture","paddock.ttft_ms":18.5,"paddock.decode_tok_s":null}]}"#
        .utf8)
    let event = try ManagerWire.decode(ActivitySnapshot.self, from: data).events[0]
    #expect(event.model == "fixture")
    #expect(event.number("paddock.ttft_ms") == 18.5)
    #expect(event.number("paddock.decode_tok_s") == nil)
    let cache = try ManagerWire.decode(
      CacheSnapshot.self,
      from: Data(
        #"{"servers":[{"port":11540,"model":null,"tier":{"lookups":0,"hits":0,"ram_ready":1048576}}]}"#
          .utf8)
    ).servers[0]
    #expect(cache.hitRate == nil)
    #expect(cache.number("ram_ready") == 1_048_576)
    #expect(cache.number("disk_ready") == nil)
  }
  @Test func liveClientProjectionContainsNoSecrets() throws {
    let setup = try ManagerWire.decode(
      LocalClientSetup.self,
      from: Data(#"{"base_url":"http://127.0.0.1:11540/v1","model":"fixture","has_key":true}"#.utf8)
    )
    #expect(setup.hasKey)
    #expect(setup.model == "fixture")
  }
  @Test func nonemptyUsageAndBenchmarkReportsDecodeTheirPublishedFields() throws {
    let usage = try ManagerWire.decode(
      UsageHistorySnapshot.self,
      from: Data(
        #"{"grain_ms":60000,"now_ms":123456,"buckets":[{"t":120000,"port":11540,"requests":2,"errors_4xx":1,"errors_5xx":0,"disconnects":0,"input_tokens":128,"output_tokens":512,"cached_tokens":7,"duration_ms_sum":5000,"spec_drafted":9,"spec_accepted":8}],"gaps":[],"generations":[],"web":[]}"#
          .utf8))
    #expect(usage.buckets[0].errors4xx == 1)
    #expect(usage.buckets[0].durationMsSum == 5000)
    let report = try ManagerWire.decode(
      BenchmarkReport.self,
      from: Data(
        #"{"id":"sample","model":"fixture","created_at_ms":123456,"concurrency":4,"prompt_words":2048,"trials":3,"warmups":1,"output_limit":128,"aggregate_output_tok_s":23.4,"wall_seconds":12.0,"ttft_median_ms":100.0,"stream_event_gap_p99_ms":null,"output_tokens":1536,"max_ctx":4096,"max_batch":4,"cache_policy":"Unique prefixes","runner_version":null,"samples":[{"input_tokens":2048,"output_tokens":128,"cached_tokens":7,"ttft_ms":100,"duration_ms":1000,"finish_reason":"length"}]}"#
          .utf8))
    #expect(report.streamEventGapP99Ms == nil)
    #expect(report.aggregateOutputTokS == 23.4)
    #expect(report.samples[0].cachedTokens == 7)
  }
}

private actor MaintenanceFixture: ManagerLoading {
  let payload: String?
  private(set) var closed = false
  private(set) var started = false
  init(payload: String?) { self.payload = payload }
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.closed }
  func maintenance(_ command: MaintenanceCommand) async throws -> MaintenanceReply {
    switch command {
    case .close:
      closed = true
      return MaintenanceReply(id: "receipt", state: "closed", payload: nil, message: nil)
    default:
      started = true
      return MaintenanceReply(
        id: "receipt", state: payload == nil ? "running" : "complete", payload: payload,
        message: nil)
    }
  }
}
