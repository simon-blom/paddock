import Foundation
import PaddockClient
import Synchronization
import Testing

@testable import PaddockConversationCore

@Suite(.serialized)
struct CompactionTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  func fixture() async throws -> NativeStudioRuntime {
    CompactionProtocol.state.withLock { $0 = .init() }
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43212", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [CompactionProtocol.self]
    let transport = try NativeConversationTransport(host: host, configuration: config)
    let runtime = NativeStudioRuntime(transport: transport) { _ in }
    try await runtime.seedContextFixture()
    return runtime
  }
  func settle(_ runtime: NativeStudioRuntime) async throws {
    for _ in 0..<500 {
      if await runtime.compactionTask == nil { return }
      try await Task.sleep(for: .milliseconds(5))
    }
    throw ConversationFailure.invalid("Compaction did not settle")
  }
  @Test func backgroundSummaryPersistsAndTheNextRequestUsesIt() async throws {
    let runtime = try await fixture()
    let before = try #require(await runtime.currentFields())
    await runtime.scheduleCompaction()
    try await settle(runtime)
    let after = try #require(await runtime.currentFields())
    #expect(after["summary"] == .string("A bounded synthetic summary"))
    #expect(after["messages"] == before["messages"])
    #expect(after["summaryModel"] == before["model"])
    // The bounded planning reserve leaves room for all four raw turns at 8K.
    // A saved summary is reused only once trimming is actually necessary.
    let untrimmed = try await runtime.requestBody(
      modelID: "cloud:ep:remote", messageID: "pending", continuing: false)
    #expect(untrimmed["input"]?.array?.count == 4)
    #expect(untrimmed["instructions"]?.string?.contains("A bounded synthetic summary") == false)
    await runtime.useSmallerContext()
    let body = try await runtime.requestBody(
      modelID: "cloud:ep:remote", messageID: "pending", continuing: false)
    #expect(body["instructions"]?.string?.contains("A bounded synthetic summary") == true)
    #expect(body["input"]?.array?.count == 2)
    let requests = CompactionProtocol.state.withLock { $0.requests }
    let summary = try #require(requests.first?["body"])
    #expect(summary["model"] == .string("remote"))
    #expect(summary["max_output_tokens"] == .number(640))
    #expect(summary["tools"] == nil && summary["previous_response_id"] == nil)
    #expect(CompactionProtocol.state.withLock { $0.saved?["summary"] } == after["summary"])
    await runtime.close()
  }
  @Test func failedSaveRollsBackOnlySummaryAndReportsFailure() async throws {
    let runtime = try await fixture()
    CompactionProtocol.state.withLock { $0.failSave = true }
    let before = try #require(await runtime.currentFields())
    await runtime.scheduleCompaction()
    try await settle(runtime)
    #expect(await runtime.currentFields() == before)
    #expect(await runtime.compactionNotice.isEmpty == false)
    await runtime.close()
  }
  @Test func timeoutCancelsTransportAndReleasesSlotForLaterSuccess() async throws {
    let runtime = try await fixture()
    CompactionProtocol.state.withLock { $0.hang = true }
    await runtime.shortCompactionDeadline()
    await runtime.scheduleCompaction()
    try await settle(runtime)
    #expect(await runtime.currentFields()?["summary"] == nil)
    #expect(CompactionProtocol.state.withLock { $0.cancelled > 0 })
    CompactionProtocol.state.withLock { $0.hang = false }
    await runtime.restoreCompactionDeadline()
    await runtime.scheduleCompaction()
    try await settle(runtime)
    #expect(await runtime.currentFields()?["summary"] != nil)
    await runtime.close()
  }
  @Test func modelSwitchCancelsSummaryAndNeverSavesItForAnotherModel() async throws {
    let runtime = try await fixture()
    CompactionProtocol.state.withLock { $0.hang = true }
    await runtime.scheduleCompaction()
    for _ in 0..<200 {
      if CompactionProtocol.state.withLock({ !$0.requests.isEmpty }) { break }
      try await Task.sleep(for: .milliseconds(5))
    }
    _ = try await runtime.command("models", ["ids": .array([.string("cloud:ep:other")])])
    #expect(await runtime.currentFields()?["model"] == .string("cloud:ep:other"))
    #expect(await runtime.currentFields()?["summary"] == nil)
    #expect(await runtime.compactionTask == nil)
    await runtime.close()
  }
  @Test func rejectedLateResultsPreserveNewerMetadataAndBranch() async throws {
    let runtime = try await fixture()
    let source = try ConversationDocument(fields: #require(await runtime.currentFields()))
    let epoch = await runtime.compactionEpoch
    await runtime.alterCoveredPrefix()
    try await runtime.acceptSummary("Stale", source: source, count: 2, epoch: epoch)
    #expect(await runtime.currentFields()?["summary"] == nil)
    #expect(await runtime.currentFields()?["title"] == .string("New title"))
    await runtime.close()
  }
  @Test func duplicateRequestsAndIncompleteSummariesDoNotMutateHistory() async throws {
    let runtime = try await fixture()
    CompactionProtocol.state.withLock { $0.incomplete = true }
    let before = await runtime.currentFields()
    await runtime.scheduleCompaction()
    await runtime.scheduleCompaction()
    try await settle(runtime)
    #expect(await runtime.currentFields() == before)
    #expect(CompactionProtocol.state.withLock { $0.requests.count } == 1)
    await runtime.close()
  }
  @Test func retryUsesOneCorrectedCapAndNeverRepeatsAStreamedTurn() async throws {
    let runtime = try await fixture()
    CompactionProtocol.state.withLock { $0.overflow = true }
    let body: O = [
      "model": .string("remote"), "stream": .bool(true), "input": .array([]),
      "max_output_tokens": .number(4000),
    ]
    try await runtime.seedReducer()
    let result = try await runtime.responseWithContextRetry(
      endpoint: .cloud("ep"), body: body,
      modelID: "cloud:ep:remote", messageID: "m3", continuing: false)
    #expect(result.status == "completed")
    #expect(
      CompactionProtocol.state.withLock { $0.requests.last?["body"]?["max_output_tokens"] }
        == .number(1192))
    #expect(
      await runtime.currentFields()?["messages"]?.array?.last?["run"]?["params"]?["maxTokens"]
        == .number(1192))
    CompactionProtocol.state.withLock { $0.overflow = true }
    await #expect(throws: (any Error).self) {
      try await runtime.responseWithContextRetry(
        endpoint: .cloud("ep"), body: body,
        modelID: "cloud:ep:remote", messageID: "m3", continuing: false)
    }
    #expect(CompactionProtocol.state.withLock { $0.requests.count } == 3)
    await runtime.close()
  }
  @Test func providerCountsCanRecoverUnknownAutomaticCapacityWithoutRepeatingAStream() async throws
  {
    let runtime = try await fixture()
    CompactionProtocol.state.withLock { $0.overflow = true }
    try await runtime.seedReducer()
    let result = try await runtime.responseWithContextRetry(
      endpoint: .cloud("ep"),
      body: ["model": .string("remote"), "stream": .bool(true), "input": .array([])],
      modelID: "cloud:ep:remote", messageID: "m3", continuing: false)
    #expect(result.status == "completed")
    #expect(CompactionProtocol.state.withLock { $0.requests.count } == 2)
    #expect(
      CompactionProtocol.state.withLock { $0.requests.last?["body"]?["max_output_tokens"] }
        == .number(1192))
    await runtime.close()
  }
}

extension NativeStudioRuntime {
  fileprivate func seedContextFixture() throws {
    document = try ContextPlanTests.document()
    draft = false
    preferences = ["pk_auto_title": .string("off")]
    models = ["remote", "other"].map { wire in
      [
        "id": .string("cloud:ep:\(wire)"), "endpoint": .string("ep"), "wireModel": .string(wire),
        "kind": .string("chat"), "status": .string("ok"), "title": .string(wire),
        "vendor": .string(""), "provider": .string("Test"),
      ]
    }
    caps = [
      "cloud:ep:remote": ["max_ctx": .number(8192)], "cloud:ep:other": ["max_ctx": .number(8192)],
    ]
  }
  fileprivate func shortCompactionDeadline() { compactionTimeout = .milliseconds(50) }
  fileprivate func useSmallerContext() { caps["cloud:ep:remote"]?["max_ctx"] = .number(4096) }
  fileprivate func restoreCompactionDeadline() { compactionTimeout = .seconds(600) }
  fileprivate func alterCoveredPrefix() {
    try? change { f in
      f["title"] = .string("New title")
      var messages = f["messages"]!.array!
      var first = messages[0].object!
      first["content"] = .array([.object(["type": .string("text"), "text": .string("Edited")])])
      messages[0] = .object(first)
      f["messages"] = .array(messages)
    }
  }
  fileprivate func seedReducer() throws { reducers["m3"] = ResponseAccumulator() }
}

private final class CompactionProtocol: URLProtocol, @unchecked Sendable {
  typealias V = ConversationValue
  typealias O = [String: V]
  struct State: Sendable {
    var requests: [O] = []
    var saved: O?
    var failSave = false, hang = false, incomplete = false, overflow = false
    var cancelled = 0
  }
  static let state = Mutex(State())
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    var bytes = request.httpBody ?? Data()
    if let stream = request.httpBodyStream {
      stream.open()
      defer { stream.close() }
      var buffer = [UInt8](repeating: 0, count: 4096)
      while stream.hasBytesAvailable {
        let n = stream.read(&buffer, maxLength: buffer.count)
        if n <= 0 { break }
        bytes.append(contentsOf: buffer.prefix(n))
      }
    }
    let body = (try? JSONDecoder().decode(V.self, from: bytes)) ?? .null
    let response: (Int, String, Data)? = Self.state.withLock { s in
      s.requests.append(["path": .string(request.url!.path), "body": body])
      if request.httpMethod == "PUT" {
        if !s.failSave { s.saved = body.object }
        return (s.failSave ? 503 : 200, "application/json", Data("{}".utf8))
      }
      if s.hang { return nil }
      if s.overflow {
        s.overflow = false
        return (
          400, "application/json",
          Data(
            #"{"error":{"message":"input length and max_tokens exceed context limit: 7000 + 4000 > 8192"}}"#
              .utf8)
        )
      }
      let status = s.incomplete ? "incomplete" : "completed"
      let terminal: O = [
        "type": .string("response.\(status)"),
        "response": .object([
          "status": .string(status),
          "output": .array([
            .object([
              "type": .string("message"),
              "content": .array([
                .object([
                  "type": .string("output_text"), "text": .string("A bounded synthetic summary"),
                ])
              ]),
            ])
          ]),
        ]),
      ]
      let json = String(decoding: try! JSONEncoder().encode(terminal), as: UTF8.self)
      return (200, "text/event-stream", Data("data: \(json)\n\n".utf8))
    }
    guard let response else { return }
    client?.urlProtocol(
      self,
      didReceive: HTTPURLResponse(
        url: request.url!, statusCode: response.0,
        httpVersion: "HTTP/1.1", headerFields: ["Content-Type": response.1])!,
      cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: response.2)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() { Self.state.withLock { $0.cancelled += 1 } }
}
