import AppKit
import Foundation
import Testing

@testable import PaddockClient
@testable import PaddockUI

@Suite("Native Studio ownership", .timeLimit(.minutes(1))) @MainActor
struct StudioChatTests {
  @Test func rejectedSendDoesNotAcceptTheDraft() async throws {
    let client = ChatFixture()
    let chat = StudioChatModel(client: client)
    let sending = Task { await chat.send("Keep this draft", runner: try runner()) }
    await client.waitForSend()
    await client.reject()
    #expect(try await sending.value == false)
    #expect(!chat.busy)
    #expect(chat.conversation == nil)
    #expect(chat.error == "Runner changed")
    await chat.shutdown()
  }
  @Test func stopBeforeReceiptCancelsAcceptedWorkAndDrainsSavedTerminal() async throws {
    _ = NSApplication.shared
    let client = ChatFixture()
    let chat = StudioChatModel(client: client)
    let sending = Task { await chat.send("Hello", runner: try runner()) }
    await client.waitForSend()
    #expect(await chat.send("Duplicate", runner: try runner()) == false)
    await chat.cancel()
    try await client.accept()
    #expect(try await sending.value)
    for _ in 0..<100 {
      if !chat.busy { break }
      try await Task.sleep(for: .milliseconds(10))
    }
    #expect(await client.cancelled)
    #expect(!chat.busy)
    #expect(chat.conversation?.messages.last?.stopped == true)
    #expect(chat.conversation?.messages.last?.streaming == false)
    await chat.shutdown()
  }
  private func runner() throws -> RunnerInfo {
    try ManagerWire.decode(
      RunnerInfo.self,
      from: Data(
        #"{"port":12345,"pid":42,"status":"ok","model":"fixture","endpoint":"http://127.0.0.1:12345"}"#
          .utf8))
  }
}
private actor ChatFixture: ManagerLoading {
  private var pending: CheckedContinuation<ChatReply, any Error>?
  private(set) var cancelled = false
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.core("Unused fixture") }
  func waitForSend() async { while pending == nil { await Task.yield() } }
  func reject() {
    pending?.resume(throwing: ManagerError.core("Runner changed"))
    pending = nil
  }
  func accept() throws {
    let reply = try decode(["stream_id": "s", "conversation": document(stopped: false)])
    pending?.resume(returning: reply)
    pending = nil
  }
  func chat(_ command: ChatCommand) async throws -> ChatReply {
    switch command.kind {
    case "send": return try await withCheckedThrowingContinuation { pending = $0 }
    case "cancel":
      cancelled = true
      return try decode(["accepted": true])
    case "poll":
      return try decode(["events": [], "done": ["conversation_id": "c", "status": "cancelled"]])
    case "load": return try decode(["conversation": document(stopped: true)])
    case "list": return try decode(["conversations": []])
    default: throw ManagerError.core("Unexpected fixture command")
    }
  }
  private func document(stopped: Bool) -> [String: Any] {
    [
      "id": "c", "title": "Hello", "model": "fixture",
      "messages": [
        ["id": "u", "role": "user", "content": [["type": "text", "text": "Hello"]]],
        [
          "id": "a", "role": "assistant", "content": [["type": "text", "text": ""]],
          "streaming": !stopped, "stopped": stopped,
        ],
      ],
    ]
  }
  private func decode(_ value: [String: Any]) throws -> ChatReply {
    try ManagerWire.decode(ChatReply.self, from: JSONSerialization.data(withJSONObject: value))
  }
}
