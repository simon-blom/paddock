import Foundation

/// Separate projections and lifetimes for the two viewer slots. Neither gets
/// the native transcript, credentials, settings or the other pane's source.
public enum NativeViewerRole: String, CaseIterable, Sendable {
  case document, graph

  public func project(_ fields: [String: ConversationValue]) -> [String: ConversationValue] {
    var result = ["conversationId": fields["conversationId"] ?? .null]
    let keys =
      self == .document
      ? ["document"] : ["graph", "graphSource", "graphHistory", "visibleGraph"]
    for key in keys { result[key] = fields[key] ?? .null }
    return result
  }
}

extension NativeStudioRuntime {
  /// Only the active branch's graph queries, never tool outputs or unrelated
  /// chat text. Match the web pane's twenty-query retention limit. A newer
  /// attachment starts a new history, even when an older graph remains in chat.
  static func graphHistory(_ document: ConversationDocument) -> [V] {
    let messages = document.activeMessages
    let start =
      messages.lastIndex {
        ($0["content"]?.array ?? []).contains { $0["type"]?.string == "graph" }
      } ?? messages.endIndex
    var remaining = 20
    var history: [V] = []
    for message in messages[start...].reversed() {
      var calls: [V] = []
      for call in (message["toolCalls"]?.array ?? []).reversed() {
        guard remaining > 0, call["serverLabel"]?.string == "graph",
          call["name"]?.string == "graph_query",
          let arguments = call["arguments"]?.string, arguments.utf8.count <= 64 * 1024,
          let parsed = try? JSONDecoder().decode(O.self, from: Data(arguments.utf8)),
          let cypher = parsed["cypher"]?.string, !cypher.isEmpty
        else { continue }
        calls.append(
          .object([
            "name": .string("graph_query"), "serverLabel": .string("graph"),
            "arguments": .string(arguments),
          ]))
        remaining -= 1
      }
      if !calls.isEmpty {
        history.append(
          .object([
            "id": message["id"] ?? .null, "role": message["role"] ?? .string("assistant"),
            "model": message["model"] ?? .string(""), "content": .array([]),
            "toolCalls": .array(calls.reversed()),
          ]))
      }
      if remaining == 0 { break }
    }
    return history.reversed()
  }
}
