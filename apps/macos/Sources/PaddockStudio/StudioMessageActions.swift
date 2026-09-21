import Foundation
import PaddockClient

/// A displayed-path token, not a copy of the conversation tree. Both sides
/// reject a stale token; only the shared store constructs or selects branches.
public struct StudioMessageTarget: Equatable, Sendable {
  public let conversationId: String
  public let leafId: String
  public let messageId: String
  public init?(transcript: StudioState.NativeTranscript, messageId: String) {
    guard transcript.available, let conversationId = transcript.conversationId,
      let leafId = transcript.leafId, transcript.messages.contains(where: { $0.id == messageId })
    else { return nil }
    self.conversationId = conversationId
    self.leafId = leafId
    self.messageId = messageId
  }
  public func payload(action: String) -> [String: StudioValue] {
    [
      "action": .string(action), "conversationId": .string(conversationId),
      "leafId": .string(leafId), "messageId": .string(messageId),
    ]
  }
}

public struct StudioMessageEdit: Equatable, Sendable {
  public let target: StudioMessageTarget
  public let originalText: String
  public var text: String
  public init(target: StudioMessageTarget, text: String) {
    self.target = target
    originalText = text
    self.text = text
  }
  public var canSubmit: Bool {
    !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && text.utf8.count <= 128 * 1024
  }
}

extension StudioWorkspace {
  public var hasMessageEdit: Bool { messageEdit != nil }

  public func beginMessageEdit(_ target: StudioMessageTarget) {
    guard ready, !busy, messageEdit == nil, let message = currentMessage(target),
      message.actions?.edit == true
    else { return }
    messageEdit = StudioMessageEdit(target: target, text: message.text)
  }
  public func cancelMessageEdit() {
    guard !messageMutation else { return }
    messageEdit = nil
  }
  private func currentMessage(_ target: StudioMessageTarget) -> StudioState.NativeTranscript
    .Message?
  {
    guard let transcript = state?.nativeTranscript, transcript.available,
      transcript.conversationId == target.conversationId, transcript.leafId == target.leafId
    else { return nil }
    return transcript.messages.first { $0.id == target.messageId }
  }
  @discardableResult public func submitMessageEdit() async -> Bool {
    guard let edit = messageEdit, edit.canSubmit else { return false }
    var payload = edit.target.payload(action: "edit")
    payload["text"] = .string(edit.text)
    payload["originalText"] = .string(edit.originalText)
    let accepted = await mutateMessage(edit.target, payload: payload)
    if accepted, messageEdit?.target == edit.target { messageEdit = nil }
    return accepted
  }
  @discardableResult public func messageAction(
    _ action: String, target: StudioMessageTarget, branch: String? = nil
  ) async -> Bool {
    guard messageEdit == nil, ["retry", "continue", "branch"].contains(action) else { return false }
    var payload = target.payload(action: action)
    if let branch { payload["targetId"] = .string(branch) }
    return await mutateMessage(target, payload: payload)
  }
  private func mutateMessage(_ target: StudioMessageTarget, payload: [String: StudioValue]) async
    -> Bool
  {
    guard ready, !busy, !uploading else { return false }
    guard currentMessage(target) != nil else {
      error = "The conversation branch changed. Your edit has been kept."
      return false
    }
    messageMutation = true
    messageNavigationAnchor = payload["action"]?.text == "branch" ? payload["targetId"]?.text : nil
    defer { messageMutation = false }
    do {
      error = nil
      let result = try await command("messageAction", payload)
      guard result["accepted"]?.boolean == true else {
        throw ManagerError.core("The message action was not accepted")
      }
      return true
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }
}
