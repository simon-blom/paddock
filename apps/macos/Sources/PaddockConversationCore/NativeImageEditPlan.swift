import Foundation

extension NativeImageGeneration {
  struct Plan: Sendable {
    let prompt: String
    let references: [O]
    let from: String?
  }

  /// Port of web Studio's referencesFor/lastUserText: current attachments win;
  /// otherwise use the last completed generated picture on the active branch.
  /// Stop at the *step*, not a lane: a faster Compare sibling must never become
  /// the other lane's input. Both lanes receive the same immutable reference.
  static func plan(document: ConversationDocument, message: O, caps: O) throws -> Plan {
    let steps = document.activeSteps
    guard
      let at = steps.firstIndex(where: {
        $0.messages.contains { $0["id"] == message["id"] }
      })
    else { throw ConversationFailure.stale }
    let history = steps.prefix(at).flatMap(\.messages)
    guard let user = history.last(where: { $0["role"]?.string == "user" }) else {
      throw ConversationFailure.invalid("Describe the picture to make")
    }
    let attached = (user["content"]?.array ?? []).compactMap(\.object).filter {
      $0["type"]?.string != "text"
    }
    try validateReferences(attached, caps: caps)
    var references = attached
    var from: String? = attached.isEmpty ? nil : "attached"
    if references.isEmpty, caps["edit"]?.bool == true {
      for prior in history.reversed()
      where prior["role"]?.string == "assistant"
        && prior["imageGen"] != nil && prior["streaming"]?.bool != true
      {
        if let picture = prior["content"]?.array?.last(where: {
          $0["type"]?.string == "image" && $0["gen"]?.object != nil
            && $0["gen"]?["preview"]?.bool != true
        })?.object {
          references = [picture]
          from = "previous"
          break
        }
      }
    }
    try validateReferences(references, caps: caps)
    let prompt =
      references.isEmpty
      ? try textPrompt(history)
      : ConversationDocument.text(user).trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else {
      throw ConversationFailure.invalid(
        references.isEmpty ? "Describe the picture to make" : "Describe how to change the picture")
    }
    return Plan(prompt: prompt, references: references, from: from)
  }

  /// Shared admission guard for composer, keyboard sends, restored turns and
  /// retries. Never discard a PDF, unavailable original or excess reference.
  static func validateReferences(_ references: [O], caps: O) throws {
    guard !references.isEmpty else { return }
    guard references.allSatisfy({ $0["type"]?.string == "image" }) else {
      throw ConversationFailure.invalid("Image editing accepts pictures, not documents or audio")
    }
    guard caps["edit"]?.bool == true else {
      throw ConversationFailure.invalid(
        "The selected image model does not support reference-image editing")
    }
    let maximum = max(1, min(10, caps["max_references"]?.integer ?? 1))
    guard references.count <= maximum else {
      throw ConversationFailure.invalid(
        "This model takes up to \(maximum) reference pictures — \(references.count) were attached")
    }
    guard !references.contains(where: { $0["unreadable"]?.bool == true }) else {
      throw ConversationFailure.invalid(
        "A reference picture is unreadable. Reattach its original file")
    }
  }
}
