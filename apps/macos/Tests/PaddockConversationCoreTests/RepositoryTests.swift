import Foundation
import Testing

@testable import PaddockConversationCore

private actor MemoryConversations: ConversationStorage {
  var doc: ConversationDocument
  var reads = 0, saves = 0
  var rejectSave = false
  init() throws {
    doc = try ConversationDocument(
      data: Data(
        #"{"id":"chat","title":"Original","updatedAt":1,"messages":[{"id":"q","parentId":null,"role":"user","content":[]},{"id":"a","parentId":"q","role":"assistant","content":[]},{"id":"b","parentId":"q","role":"assistant","content":[]}],"leafId":"a","future":"keep"}"#
          .utf8))
  }
  func loadConversation(_ id: String) async throws -> ConversationDocument {
    reads += 1
    return doc
  }
  func saveConversation(_ document: ConversationDocument) async throws {
    // Force actor reentrancy: two simultaneous edits must not both read the
    // old document before either durable acknowledgment has arrived.
    try await Task.sleep(for: .milliseconds(15))
    if rejectSave { throw ConversationFailure.http(500) }
    doc = document
    saves += 1
  }
  func reject(_ value: Bool) { rejectSave = value }
}

@Suite("Native conversation persistence")
struct RepositoryTests {
  @Test func concurrentMetadataEditsSerializeAndKeepMessages() async throws {
    let storage = try MemoryConversations()
    let repository = ConversationRepository(storage: storage)
    async let rename = repository.edit("chat", .rename("Renamed"))
    async let pin = repository.edit("chat", .pin(true))
    _ = try await (rename, pin)
    let result = try await repository.load("chat")
    #expect(result.title == "Renamed")
    #expect(result.fields["pinned"]?.bool == true)
    #expect(result.fields["future"]?.string == "keep")
    #expect(result.fields["updatedAt"]?.integer == 1)
    #expect(result.messages.count == 3)
    #expect(await storage.saves == 2)
  }
  @Test func rejectedSaveDoesNotPublishOrLoseFollowingEdits() async throws {
    let storage = try MemoryConversations()
    let repository = ConversationRepository(storage: storage)
    await storage.reject(true)
    await #expect(throws: ConversationFailure.http(500)) {
      try await repository.edit("chat", .rename("Unstored"))
    }
    #expect(await storage.doc.title == "Original")
    await storage.reject(false)
    let next = try await repository.edit("chat", .pin(true))
    #expect(next.title == "Original")
    #expect(next.fields["pinned"]?.bool == true)
    await repository.close()
    await #expect(throws: ConversationFailure.closed) {
      try await repository.edit("chat", .rename("Closed"))
    }
  }
  @Test func staleBranchCannotChangeAnotherPath() async throws {
    let storage = try MemoryConversations()
    let repository = ConversationRepository(storage: storage)
    let changed = try await repository.edit(
      "chat", .branch(expectedLeaf: "a", message: "a", target: "b"))
    #expect(changed.leafID == "b")
    await #expect(throws: ConversationFailure.stale) {
      try await repository.edit("chat", .branch(expectedLeaf: "a", message: "a", target: "b"))
    }
    #expect(await storage.saves == 1)
  }
}
