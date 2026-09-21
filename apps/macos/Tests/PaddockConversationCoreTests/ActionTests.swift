import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Native message action eligibility")
struct ActionTests {
  private func document() throws -> ConversationDocument {
    try ConversationDocument(
      data: Data(
        #"{"id":"c","leafId":"a","messages":[{"id":"u","parentId":null,"role":"user","content":[{"type":"text","text":"Original"},{"type":"file","attachmentId":"pdf","pageRange":"3-8","future":"keep"}]},{"id":"a","parentId":"u","role":"assistant","incomplete":"length","content":[{"type":"text","text":"Answer"}]}]}"#
          .utf8))
  }
  @Test func editRetainsTheStoredAttachmentAndOnlyReplacesText() throws {
    let doc = try document()
    let target = ConversationActionTarget(conversationID: "c", leafID: "a", messageID: "u")
    let edit = try doc.resolve(target, action: .edit(text: " Revised ", originalText: "Original"))
    #expect(edit.parts.first == doc.messages[0]["content"]?.array?[1])
    #expect(edit.parts.last?["text"]?.string == "Revised")
    #expect(throws: ConversationFailure.stale) {
      try doc.resolve(target, action: .edit(text: "Revised", originalText: "Stale"))
    }
    #expect(throws: ConversationFailure.stale) {
      try doc.resolve(
        .init(conversationID: "other", leafID: "a", messageID: "u"),
        action: .edit(text: "Revised", originalText: "Original"))
    }
  }
  @Test func continuationIsOnlyTheLengthLimitedSingleModelTail() throws {
    let doc = try document()
    let target = ConversationActionTarget(conversationID: "c", leafID: "a", messageID: "a")
    _ = try doc.resolve(target, action: .retry)
    _ = try doc.resolve(target, action: .continueReply)
    var fields = doc.fields
    fields["compareModels"] = .array([.string("A"), .string("B")])
    let comparison = try ConversationDocument(fields: fields)
    #expect(throws: ConversationFailure.stale) {
      try comparison.resolve(target, action: .continueReply)
    }
    #expect(throws: ConversationFailure.stale) { try comparison.resolve(target, action: .retry) }
  }
}
