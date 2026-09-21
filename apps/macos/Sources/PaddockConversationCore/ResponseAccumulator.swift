import Foundation

/// Native semantic reducer. Keeps the full terminal object, including output
/// kinds not yet projected into native controls. It never executes tool code.
public struct ResponseAccumulator: Sendable {
  public typealias Value = ConversationValue
  private struct Part: Hashable, Comparable, Sendable {
    let output: Int, content: Int
    static func < (lhs: Self, rhs: Self) -> Bool {
      lhs.output == rhs.output ? lhs.content < rhs.content : lhs.output < rhs.output
    }
  }
  private var textParts = [Part: String](), reasoningParts = [Part: String]()
  private var usedBytes = 0
  private var sequence: Int?
  public private(set) var status: String?
  public private(set) var terminal: [String: Value]?
  public private(set) var failure: String?
  public let maximumBytes: Int
  public init(maximumBytes: Int = 16 * 1024 * 1024) { self.maximumBytes = max(1, maximumBytes) }
  public var text: String {
    textParts.keys.sorted().compactMap { textParts[$0] }.joined(separator: "\n")
  }
  public var reasoning: String {
    reasoningParts.keys.sorted().compactMap { reasoningParts[$0] }.joined(separator: "\n")
  }

  /// Returns the raw semantic event for tool/search/usage adapters. Unknown
  /// events are not fabricated as text and are not interpreted as completion.
  @discardableResult public mutating func apply(_ data: String) throws -> [String: Value]? {
    if data == "[DONE]" { return nil }
    guard status == nil else {
      throw ConversationFailure.invalid("Event after response completion")
    }
    guard data.utf8.count <= maximumBytes else { throw ConversationFailure.tooLarge }
    let event = try JSONDecoder().decode([String: Value].self, from: Data(data.utf8))
    try apply(event)
    return event
  }
  public mutating func apply(_ event: [String: Value]) throws {
    guard status == nil else {
      throw ConversationFailure.invalid("Event after response completion")
    }
    guard let kind = event["type"]?.string else {
      throw ConversationFailure.invalid("Response event has no type")
    }
    if let value = event["sequence_number"] {
      guard let next = value.integer, next >= 0, sequence == nil || next > sequence! else {
        throw ConversationFailure.invalid("Response sequence is not increasing")
      }
      sequence = next
    }
    switch kind {
    case "response.output_text.delta", "response.refusal.delta", "response.reasoning_text.delta",
      "response.reasoning_summary_text.delta", "response.reasoning.delta":
      guard let delta = event["delta"]?.string else {
        throw ConversationFailure.invalid("Response delta is not text")
      }
      let output = event["output_index"]?.integer ?? 0
      let content = event["content_index"]?.integer ?? event["summary_index"]?.integer ?? 0
      guard output >= 0, content >= 0 else {
        throw ConversationFailure.invalid("Negative response part index")
      }
      guard delta.utf8.count <= maximumBytes - usedBytes else { throw ConversationFailure.tooLarge }
      usedBytes += delta.utf8.count
      let key = Part(output: output, content: content)
      if kind.contains("reasoning") {
        reasoningParts[key, default: ""].append(delta)
      } else {
        textParts[key, default: ""].append(delta)
      }
    case "response.completed", "response.incomplete", "response.failed":
      guard let response = event["response"]?.object else {
        throw ConversationFailure.invalid("Terminal event has no response")
      }
      let result = String(kind.dropFirst("response.".count))
      guard response["status"] == nil || response["status"]?.string == result else {
        throw ConversationFailure.invalid("Terminal response status disagrees with event")
      }
      var finalText = [Part: String]()
      var finalReasoning = [Part: String]()
      var hasMessage = false
      var hasReasoning = false
      var bytes = 0
      for (i, item) in (response["output"]?.array ?? []).enumerated() {
        let reasoning = item["type"]?.string == "reasoning"
        let message = item["type"]?.string == "message"
        guard reasoning || message else { continue }  // Retained in terminal, not discarded.
        hasReasoning = hasReasoning || reasoning
        hasMessage = hasMessage || message
        // Summary and full reasoning are alternative renderings of the same
        // item. Prefer its full content, otherwise its summary.
        let content = item["content"]?.array ?? []
        let parts = content.isEmpty && reasoning ? item["summary"]?.array ?? [] : content
        for (j, part) in parts.enumerated() {
          guard let text = part["text"]?.string ?? part["refusal"]?.string else { continue }
          guard text.utf8.count <= maximumBytes - bytes else { throw ConversationFailure.tooLarge }
          bytes += text.utf8.count
          let key = Part(output: i, content: j)
          if reasoning { finalReasoning[key] = text } else { finalText[key] = text }
        }
      }
      // Explicitly empty output text is authoritative; omitted reasoning on
      // local runners must not erase reasoning which was actually streamed.
      if hasMessage { textParts = finalText }
      if hasReasoning { reasoningParts = finalReasoning }
      terminal = response
      status = result
      failure = response["error"]?["message"]?.string
    case "error":
      throw ConversationFailure.invalid(
        event["message"]?.string ?? event["error"]?["message"]?.string ?? "Response stream failed")
    default: break
    }
  }
  public func requireTerminal() throws {
    guard status != nil else { throw ConversationFailure.interrupted }
  }
}
