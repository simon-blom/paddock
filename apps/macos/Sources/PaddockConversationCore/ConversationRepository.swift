import Foundation

public protocol ConversationStorage: Sendable {
  func loadConversation(_ id: String) async throws -> ConversationDocument
  func saveConversation(_ document: ConversationDocument) async throws
}

/// Native metadata/branch transactions. A list row is never a writable
/// document. Every edit reads the complete document, serializes behind earlier
/// edits to the same chat, then waits for the Rust/SQLite acknowledgment.
public actor ConversationRepository {
  private let storage: any ConversationStorage
  private struct Pending {
    let id: UUID
    let task: Task<ConversationDocument, any Error>
  }
  private var writes = [String: Pending]()
  private var closed = false
  public init(storage: any ConversationStorage) { self.storage = storage }

  public enum Edit: Sendable {
    case rename(String)
    case pin(Bool)
    case branch(expectedLeaf: String, message: String, target: String)
  }
  public func load(_ id: String) async throws -> ConversationDocument {
    guard !closed else { throw ConversationFailure.closed }
    if let pending = writes[id] { _ = try? await pending.task.value }
    var doc = try await storage.loadConversation(id)
    doc.migrateTree()
    return doc
  }
  public func edit(_ id: String, _ edit: Edit) async throws -> ConversationDocument {
    guard !closed else { throw ConversationFailure.closed }
    guard ConversationDocument.validID(id) else {
      throw ConversationFailure.invalid("Invalid conversation identity")
    }
    // Before admission only: accepted writes must return their receipt.
    try Task.checkCancellation()
    let previous = writes[id]?.task
    let storage = storage
    let ticket = UUID()
    let task = Task<ConversationDocument, any Error> {
      if let previous { _ = try? await previous.value }
      var doc = try await storage.loadConversation(id)
      guard doc.id == id else {
        throw ConversationFailure.invalid("Conversation identity mismatch")
      }
      doc.migrateTree()
      switch edit {
      case .rename(let title): try doc.rename(title)
      case .pin(let pinned): doc.setPinned(pinned)
      case .branch(let expectedLeaf, let message, let target):
        let resolved = try doc.resolve(
          .init(conversationID: id, leafID: expectedLeaf, messageID: message),
          action: .branch(targetID: target))
        guard doc.stepSibling(of: message, delta: resolved.branchDelta) else {
          throw ConversationFailure.stale
        }
      }
      try await storage.saveConversation(doc)
      return doc
    }
    writes[id] = Pending(id: ticket, task: task)
    defer { if writes[id]?.id == ticket { writes[id] = nil } }
    return try await task.value
  }
  public func close() async {
    closed = true
    let admitted = writes.values.map(\.task)
    for task in admitted { _ = try? await task.value }
  }
}
