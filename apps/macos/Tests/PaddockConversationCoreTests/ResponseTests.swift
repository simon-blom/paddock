import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Native response decoding")
struct ResponseTests {
  @Test func everyByteBoundaryCRLFAndUnicode() throws {
    let raw = Data(
      "\u{feff}: comment\r\nevent: response\r\ndata: {\"type\":\"x\",\r\ndata: \"text\":\"å🦊\"}\r\n\r\ndata: second\r\rdata: third\n\n"
        .utf8)
    for size in 1...raw.count {
      var parser = ServerSentEvents()
      var events = [ServerSentEvents.Event]()
      for offset in stride(from: 0, to: raw.count, by: size) {
        events += try parser.push(raw.subdata(in: offset..<min(raw.count, offset + size)))
      }
      #expect(events.map(\.data) == ["{\"type\":\"x\",\n\"text\":\"å🦊\"}", "second", "third"])
      #expect(events.map(\.name) == ["response", "message", "message"])
    }
  }
  @Test func boundsAndEOFNeverMeanSuccess() throws {
    var parser = ServerSentEvents(maximumBytes: 32)
    #expect(throws: ConversationFailure.tooLarge) {
      try parser.push(Data(repeating: 65, count: 33))
    }
    var invalid = ServerSentEvents()
    #expect(throws: (any Error).self) { try invalid.push(Data([0xff, 10])) }
    var incomplete = ServerSentEvents()
    let pending = try incomplete.push(Data("data: partial".utf8))
    #expect(pending.isEmpty)
    incomplete.finish()
    var response = ResponseAccumulator()
    try response.apply("[DONE]")
    #expect(throws: ConversationFailure.interrupted) { try response.requireTerminal() }
  }
  @Test func terminalCorrectsMissingTailAndKeepsToolsAndUsage() throws {
    var r = ResponseAccumulator()
    try r.apply(#"{"type":"response.output_text.delta","sequence_number":1,"delta":"Partial"}"#)
    try r.apply(#"{"type":"response.reasoning_text.delta","sequence_number":2,"delta":"Thought"}"#)
    let event = try r.apply(
      #"{"type":"response.completed","sequence_number":3,"response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Complete tail 🦊"}]},{"type":"mcp_call","id":"call","output":"keep"},{"type":"future","data":[1,2]}],"usage":{"output_tokens":10},"future":"keep"}}"#
    )
    #expect(event?["type"]?.string == "response.completed")
    #expect(r.text == "Complete tail 🦊")
    #expect(r.reasoning == "Thought")
    #expect(r.terminal?["output"]?.array?.count == 3)
    #expect(r.terminal?["future"]?.string == "keep")
    #expect(r.terminal?["usage"]?["output_tokens"]?.integer == 10)
    try r.requireTerminal()
    #expect(throws: (any Error).self) {
      try r.apply(#"{"type":"response.output_text.delta","delta":"late"}"#)
    }
  }
  @Test func emptyTerminalTextIsAuthoritativeAndReasoningNotDuplicated() throws {
    var r = ResponseAccumulator()
    try r.apply(#"{"type":"response.output_text.delta","delta":"discard"}"#)
    try r.apply(
      #"{"type":"response.completed","response":{"output":[{"type":"message","content":[]},{"type":"reasoning","content":[{"text":"full"}],"summary":[{"text":"short"}]}]}}"#
    )
    #expect(r.text.isEmpty)
    #expect(r.reasoning == "full")
  }
  @Test func sequenceFailureAndBudgetAreExplicit() throws {
    var r = ResponseAccumulator(maximumBytes: 1000)
    try r.apply(#"{"type":"response.created","sequence_number":2}"#)
    #expect(throws: (any Error).self) {
      try r.apply(#"{"type":"response.created","sequence_number":2}"#)
    }
    #expect(throws: (any Error).self) {
      try r.apply(#"{"type":"response.completed","response":{"status":"failed"}}"#)
    }
    #expect(r.status == nil)
    let delta: [String: ConversationValue] = [
      "type": .string("response.output_text.delta"),
      "delta": .string(String(repeating: "x", count: 800)),
    ]
    let json = String(decoding: try JSONEncoder().encode(delta), as: UTF8.self)
    try r.apply(json)
    #expect(throws: ConversationFailure.tooLarge) { try r.apply(json) }
    #expect(r.text.count == 800)
  }
}
