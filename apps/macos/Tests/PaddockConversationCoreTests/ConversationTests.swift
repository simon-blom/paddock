import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Native conversation document and web tree parity")
struct ConversationTests {
  @Test func sharedTreeFixtures() throws {
    let url = try #require(
      Bundle.module.url(forResource: "tree-parity", withExtension: "json", subdirectory: "Fixtures")
    )
    let fixtures = try JSONDecoder().decode(
      [[String: ConversationValue]].self, from: Data(contentsOf: url))
    for fixture in fixtures {
      var doc = try ConversationDocument(fields: #require(fixture["document"]?.object))
      for operation in try #require(fixture["operations"]?.array) {
        switch operation["kind"]?.string {
        case "migrate":
          let changed = doc.migrateTree()
          if let expected = operation["changed"]?.bool { #expect(changed == expected) }
        case "sibling":
          let changed = doc.stepSibling(
            of: operation["id"]!.string!, delta: operation["delta"]!.integer!)
          #expect(changed)
        case "delete":
          let removed = doc.deleteSubtree(operation["id"]!.string!)
          #expect(removed == Set(operation["removed"]!.array!.compactMap(\.string)))
        default: Issue.record("Unknown fixture operation")
        }
        #expect(
          doc.activeMessages.compactMap { $0["id"]?.string }
            == operation["path"]!.array!.compactMap(\.string))
        #expect(doc.leafID == operation["leaf"]?.string)
        #expect(try ConversationDocument(data: doc.encoded()) == doc)
      }
    }
  }
  @Test func metadataNeverRewritesUnknownFieldsOrOriginals() throws {
    let data = Data(
      #"{"id":"chat","title":"Old","messages":[{"id":"u","role":"user","content":[{"type":"file","attachmentId":"saved-pdf","pages":400,"future":{"keep":"yes"}}]}],"future":{"integer":9007199254740993,"decimal":1.125,"null":null,"text":"🦊\u0000"}}"#
        .utf8)
    var doc = try ConversationDocument(data: data)
    let unknown = doc.fields["future"]
    let parts = doc.messages[0]["content"]
    doc.migrateTree()
    try doc.rename("  New title  ")
    doc.setPinned(true)
    let reopened = try ConversationDocument(data: doc.encoded())
    #expect(reopened.fields["future"] == unknown)
    #expect(reopened.messages[0]["content"] == parts)
    #expect(reopened.title == "New title")
    #expect(reopened.fields["future"]?["integer"]?.integer == 9_007_199_254_740_993)
  }
  @Test func rejectAmbiguousDocumentsInsteadOfDroppingMessages() {
    for raw in [
      #"{"id":"c","messages":[{"id":"same","role":"user","content":[]},{"id":"same","role":"assistant","content":[]}]}"#,
      #"{"id":"../outside","messages":[]}"#,
      #"{"id":"c","messages":[{"id":"u","role":"user","content":[],"parentId":4}]}"#,
    ] { #expect(throws: (any Error).self) { try ConversationDocument(data: Data(raw.utf8)) } }
  }
  @Test func largeTreeIsIterative() throws {
    let messages: [ConversationValue] = (0..<10000).map { i in
      .object([
        "id": .string("m\(i)"), "role": .string(i % 2 == 0 ? "user" : "assistant"),
        "content": .array([]),
      ])
    }
    var doc = try ConversationDocument(fields: [
      "id": .string("large"), "messages": .array(messages),
    ])
    let first = doc.migrateTree()
    let second = doc.migrateTree()
    #expect(first)
    #expect(!second)
    #expect(doc.activeMessages.count == 10000)
    let removed = doc.deleteSubtree("m0")
    #expect(removed.count == 10000)
    #expect(doc.activeMessages.isEmpty)
  }
}
