import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Shared SQLite Reads contract")
struct ReadHistoryTests {
  @Test func webQuestionAndChoiceOrderSurviveSwiftRoundTrip() throws {
    let web =
      #"{"id":"web","title":"Ticket","runs":[{"state":"Full text","samples":3,"questions":{"z":{"type":"choice","criteria":{"zebra":"Z","apple":"A"}},"a":{"type":"noul"}},"response":{},"ms":1.5}]}"#
    let first = try ReadHistoryDocument(json: web)
    let second = try ReadHistoryDocument(json: first.json)
    let draft = try ReadHistoryDocument.draft(#require(second.runs.first))
    #expect(draft.state == "Full text" && draft.samples == 3)
    #expect(draft.questions.map(\.questionID) == ["z", "a"])
    #expect(draft.questions[0].options.map(\.name) == ["zebra", "apple"])
  }
  @Test func legacyResultDoesNotInventItsOriginalInput() throws {
    let doc = try ReadHistoryDocument(
      json:
        #"{"id":"legacy","runs":[{"excerpt":"Not the full input","state":"","stateMissing":true,"questions":{"q":{"type":"noul"}},"samples":"auto"}]}"#
    )
    #expect(try ReadHistoryDocument.draft(#require(doc.runs.first)).state.isEmpty)
    #expect(doc.runs.first?["stateMissing"] == .bool(true))
  }
  @Test func invalidOrderingCannotSilentlyDropQuestions() throws {
    let doc = try ReadHistoryDocument(
      json: #"{"id":"bad","runs":[{"questions":{"q":{"type":"noul"}},"questionOrder":[]}]}"#)
    #expect(throws: ConversationFailure.self) {
      try ReadHistoryDocument.draft(#require(doc.runs.first))
    }
  }
}
