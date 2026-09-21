import Foundation

extension NativeStudioRuntime {
  public func beginLive() async throws -> [String: String] {
    guard let before = document, tasks.isEmpty else { throw ConversationFailure.stale }
    var fields = before.fields
    var messages = before.messages
    let userID = UUID().uuidString
    let group = selected.count > 1 ? UUID().uuidString : nil
    messages.append([
      "id": .string(userID), "parentId": before.tipID.map(V.string) ?? .null,
      "role": .string("user"), "content": .array([]), "audioPending": .bool(true),
      "createdAt": Self.now,
    ])
    var mapping: [String: String] = [:]
    var leaf = ""
    for model in selected {
      let id = UUID().uuidString
      if leaf.isEmpty { leaf = id }
      var row: O = [
        "id": .string(id), "parentId": .string(userID), "role": .string("assistant"),
        "model": .string(model), "content": .array([]), "streaming": .bool(true),
        "createdAt": Self.now, "transcript": .object([:]),
      ]
      row["group"] = group.map(V.string)
      messages.append(row)
      mapping[model] = id
    }
    fields["messages"] = .array(messages.map(V.object))
    fields["leafId"] = .string(leaf)
    fields["updatedAt"] = Self.now
    let next = try ConversationDocument(fields: fields)
    try await save(next)
    document = next
    draft = false
    return mapping
  }
  public func liveUpdate(messageID: String, text: String, transcript: O) throws {
    guard text.utf8.count <= 512 * 1024 else { throw ConversationFailure.tooLarge }
    try updateMessage(messageID) {
      $0["content"] = .array([.object(["type": .string("text"), "text": .string(text)])])
      $0["transcript"] = .object(transcript)
    }
    schedulePublish()
  }
  public func liveFailure(messageID: String, error: String) throws {
    try updateMessage(messageID) {
      $0["error"] = .string(error)
      $0["streaming"] = .bool(false)
    }
    schedulePublish()
  }
  public func finishLive(messages: [String], clip: O?, failure: String? = nil) async throws {
    var parent: V?
    for id in messages {
      try updateMessage(id) { row in
        parent = row["parentId"]
        row["streaming"] = .bool(false)
        if let failure { row["error"] = .string(failure) }
      }
    }
    if let id = parent?.string {
      try updateMessage(id) { row in
        row["audioPending"] = nil
        if let clip {
          row["content"] = .array([.object(clip)])
        } else {
          row["content"] = .array([
            .object(["type": .string("text"), "text": .string("[Recording unavailable]")])
          ])
        }
      }
    }
    // Store the original before optional enrichment; a failing alignment must
    // never lose the successful live transcript or its recording.
    try await persist()
    schedulePublish()
    if failure == nil, let clip {
      for id in messages
      where document?.messages.first(where: { $0["id"]?.string == id })?["error"]?.string == nil {
        try await enrichSpeech(messageID: id, clip: clip)
      }
      try await persist()
      schedulePublish()
    }
  }
}
