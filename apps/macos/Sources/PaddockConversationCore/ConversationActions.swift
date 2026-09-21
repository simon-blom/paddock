import Foundation

public struct ConversationMessageControls: Sendable, Equatable {
  public struct Branch: Sendable, Equatable {
    public let index: Int, count: Int
    public let previous: String?, next: String?
  }
  public let edit: Bool, retry: Bool, continueReply: Bool
  public let branch: Branch?
}

/// Binds an action to exactly the document/path which the native UI displayed.
/// An attachment, parent ID or model URL is never accepted from an edit form.
public struct ConversationActionTarget: Sendable {
  public let conversationID: String, leafID: String, messageID: String
  public init(conversationID: String, leafID: String, messageID: String) {
    self.conversationID = conversationID
    self.leafID = leafID
    self.messageID = messageID
  }
}
public enum ConversationMessageAction: Sendable {
  case edit(text: String, originalText: String)
  case retry, continueReply
  case branch(targetID: String)
}
public struct ResolvedConversationAction: Sendable {
  public let message: ConversationDocument.Object
  public let parts: [ConversationValue]
  public let branchDelta: Int
}

extension ConversationDocument {
  /// Port of native-workspace/message-actions.ts. Action eligibility is not a
  /// guess from the currently selected model or flat message array.
  public func resolve(_ target: ConversationActionTarget, action: ConversationMessageAction) throws
    -> ResolvedConversationAction
  {
    guard target.conversationID == id, target.leafID == leafID,
      Self.validID(target.messageID),
      let message = activeMessages.first(where: { $0["id"]?.string == target.messageID }),
      message["streaming"]?.bool != true,
      let controls = messageControls[target.messageID]
    else { throw ConversationFailure.stale }
    switch action {
    case .edit(let raw, let original):
      guard controls.edit, raw.utf16.count <= 128 * 1024, original.utf16.count <= 128 * 1024,
        original == Self.text(message)
      else { throw ConversationFailure.stale }
      let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !text.isEmpty else {
        throw ConversationFailure.invalid("Write a message before sending the edit")
      }
      let attachments = (message["content"]?.array ?? []).filter { $0["type"]?.string != "text" }
      return .init(
        message: message,
        parts: attachments + [.object(["type": .string("text"), "text": .string(text)])],
        branchDelta: 0)
    case .branch(let targetID):
      guard let branch = controls.branch, let siblings = siblings(of: target.messageID),
        [branch.previous, branch.next].contains(targetID),
        let next = siblings.steps.firstIndex(where: { $0.anchorID == targetID }),
        abs(next - siblings.index) == 1
      else { throw ConversationFailure.stale }
      return .init(message: message, parts: [], branchDelta: next - siblings.index)
    case .retry:
      guard controls.retry else { throw ConversationFailure.stale }
    case .continueReply:
      guard controls.continueReply else { throw ConversationFailure.stale }
    }
    return .init(message: message, parts: [], branchDelta: 0)
  }
}
