import Foundation
import PaddockClient
import Testing

@testable import PaddockConversationCore

@Suite("Independent native viewers", .serialized)
struct ViewerProjectionTests {
  typealias V = ConversationValue
  typealias O = [String: V]

  static let pdf: V = .object([
    "type": .string("file"), "mime": .string("application/pdf"),
    "attachmentId": .string("pdf-file"), "name": .string("Paper.pdf"),
  ])
  static let graph: V = .object([
    "type": .string("graph"), "attachmentId": .string("graph-file"), "name": .string("Graph.tvdb"),
  ])
  static func message(_ id: String, parent: String? = nil, content: [V] = [], calls: [V] = []) -> O
  {
    [
      "id": .string(id), "parentId": parent.map(V.string) ?? .null,
      "role": .string("user"), "content": .array(content), "toolCalls": .array(calls),
      "model": .string("qwen"),
    ]
  }
  static func call(_ arguments: String, server: String = "graph") -> V {
    .object([
      "name": .string("graph_query"), "serverLabel": .string(server),
      "arguments": .string(arguments), "output": .string("MUST NOT CROSS THE VIEWER BRIDGE"),
    ])
  }
  static func document(_ messages: [O], leaf: String) throws -> ConversationDocument {
    try .init(fields: [
      "id": .string("chat"), "messages": .array(messages.map(V.object)), "leafId": .string(leaf),
    ])
  }
  @Test func projectionsExcludeOtherPaneAndUnrelatedData() {
    let fields: O = [
      "conversationId": .string("chat"), "document": .object(["id": .string("pdf")]),
      "graph": .object(["id": .string("graph")]), "graphSource": Self.graph,
      "graphHistory": .array([]), "visibleGraph": .bool(true),
      "credentials": .string("never"), "transcript": .string("never"),
    ]
    let left = NativeViewerRole.document.project(fields)
    let right = NativeViewerRole.graph.project(fields)
    #expect(Set(left.keys) == ["conversationId", "document"])
    #expect(
      Set(right.keys) == ["conversationId", "graph", "graphSource", "graphHistory", "visibleGraph"])
    #expect(left["document"] == fields["document"])
    #expect(right["graphSource"] == Self.graph)
  }
  @Test func historyIsBoundedActiveBranchOnlyAndContainsNoToolOutput() throws {
    var messages = [Self.message("root", content: [Self.graph])]
    for i in 0..<25 {
      messages.append(
        Self.message(
          "m\(i)", parent: i == 0 ? "root" : "m\(i - 1)",
          content: [.object(["type": .string("text"), "text": .string("PRIVATE TRANSCRIPT")])],
          calls: [Self.call("{\"cypher\":\"RETURN \(i)\"}")]))
    }
    messages.append(
      Self.message("sibling", parent: "root", calls: [Self.call("{\"cypher\":\"RETURN 999\"}")]))
    let history = NativeStudioRuntime.graphHistory(try Self.document(messages, leaf: "m24"))
    #expect(history.count == 20)
    #expect(history.first?["id"] == .string("m5"))
    #expect(history.last?["id"] == .string("m24"))
    let json = String(decoding: try JSONEncoder().encode(history), as: UTF8.self)
    #expect(!json.contains("PRIVATE") && !json.contains("MUST NOT") && !json.contains("999"))
    #expect(history.allSatisfy { $0["model"] == .string("qwen") })
  }
  @Test func malformedQueriesAndPreviousAttachmentHistoryAreExcluded() throws {
    let messages = [
      Self.message("old", content: [Self.graph], calls: [Self.call("{\"cypher\":\"RETURN 0\"}")]),
      Self.message("new", parent: "old", content: [Self.graph]),
      Self.message(
        "answer", parent: "new",
        calls: [
          Self.call("{\"cypher\":\"RETURN 1\"}"), Self.call("{"), Self.call("{\"cypher\":23}"),
          Self.call("{\"cypher\":\"\"}"),
          Self.call("{\"cypher\":\"RETURN 2\"}", server: "unrelated"),
          Self.call("{\"cypher\":\"\(String(repeating: "🦊", count: 20000))\"}"),
        ]),
    ]
    let history = NativeStudioRuntime.graphHistory(try Self.document(messages, leaf: "answer"))
    #expect(history.count == 1)
    #expect(history.first?["toolCalls"]?.array?.count == 1)
    #expect(
      NativeStudioRuntime.graphHistory(try Self.document([Self.message("empty")], leaf: "empty"))
        .isEmpty)
  }
  @Test func openingAndClosingEachPaneDoesNotDismissTheOther() async throws {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in
    }
    try await runtime.viewerFixture()
    _ = try await runtime.command("graphPanel", ["open": .bool(true)])
    _ = try await runtime.command(
      "openDocument", ["messageId": .string("source"), "attachmentId": .string("pdf-file")])
    var state = await runtime.viewerState()
    #expect(state["visibleGraph"] == .bool(true) && state["document"] != .null)
    #expect(
      Set(state["document"]?.object?.keys.map { $0 } ?? []) == [
        "id", "title", "messages", "leafId", "activeDocId",
      ])
    _ = try await runtime.command("closePreview")
    state = await runtime.viewerState()
    #expect(state["visibleGraph"] == .bool(true) && state["document"] == .null)
    _ = try await runtime.command(
      "openDocument", ["messageId": .string("source"), "attachmentId": .string("pdf-file")])
    _ = try await runtime.command("graphPanel", ["open": .bool(false)])
    state = await runtime.viewerState()
    #expect(state["visibleGraph"] == .bool(false) && state["document"] != .null)
    _ = try await runtime.command("graphPanel", ["open": .bool(true)])
    state = await runtime.viewerState()
    #expect(state["visibleGraph"] == .bool(true) && state["document"] != .null)
    _ = try await runtime.command("newChat")
    state = await runtime.viewerState()
    #expect(state["visibleGraph"] == .bool(false) && state["document"] == .null)
    #expect(state["graphHistory"] == .array([]))
    await runtime.close()
  }
  @Test func closingOrNavigatingWhileGraphContentLoadsCannotReopenTheOldPane() async throws {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    for command in ["graphPanel", "newChat"] {
      let configuration = URLSessionConfiguration.ephemeral
      configuration.protocolClasses = [DelayedViewerProtocol.self]
      let runtime = NativeStudioRuntime(
        transport: try NativeConversationTransport(host: host, configuration: configuration)
      ) { _ in }
      try await runtime.viewerFixture()
      let pending = Task { try await runtime.command("graphArtifact", ["id": .string("artifact")]) }
      for _ in 0..<500 {
        if DelayedViewerProtocol.hasPending { break }
        try await Task.sleep(for: .milliseconds(10))
      }
      try #require(DelayedViewerProtocol.hasPending)
      _ = try await runtime.command(command, ["open": .bool(false)])
      DelayedViewerProtocol.complete()
      await #expect(throws: ConversationFailure.self) { try await pending.value }
      let state = await runtime.viewerState()
      #expect(state["graph"] == .null && state["visibleGraph"] == .bool(false))
      await runtime.close()
    }
  }
}

extension NativeStudioRuntime {
  fileprivate func viewerFixture() throws {
    document = try ViewerProjectionTests.document(
      [
        ViewerProjectionTests.message(
          "source", content: [ViewerProjectionTests.pdf, ViewerProjectionTests.graph])
      ], leaf: "source")
    draft = true  // In-memory fixture: no HTTP or user library.
    artifacts = [.object(["id": .string("artifact"), "kind": .string("graph")])]
  }
}

private final class DelayedViewerProtocol: URLProtocol, @unchecked Sendable {
  private final class Pending: @unchecked Sendable {
    let lock = NSLock()
    var value: DelayedViewerProtocol?
  }
  private static let pending = Pending()
  static var hasPending: Bool { pending.lock.withLock { pending.value != nil } }
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    Self.pending.lock.withLock { Self.pending.value = self }
  }
  override func stopLoading() {
    Self.pending.lock.withLock { if Self.pending.value === self { Self.pending.value = nil } }
  }
  static func complete() {
    let request = pending.lock.withLock { () -> DelayedViewerProtocol? in
      defer { pending.value = nil }
      return pending.value
    }
    guard let request else { return }
    let response = HTTPURLResponse(
      url: request.request.url!, statusCode: 200, httpVersion: "HTTP/1.1",
      headerFields: ["Content-Type": "text/plain"])!
    request.client?.urlProtocol(request, didReceive: response, cacheStoragePolicy: .notAllowed)
    request.client?.urlProtocol(request, didLoad: Data("CREATE (:Person)".utf8))
    request.client?.urlProtocolDidFinishLoading(request)
  }
}
