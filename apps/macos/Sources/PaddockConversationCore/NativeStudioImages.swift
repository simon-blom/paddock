import Foundation

extension NativeStudioRuntime {
  func generateImages(modelID: String, messageID: String) async throws {
    guard let document,
      let message = document.messages.first(where: { $0["id"]?.string == messageID }),
      let cap = capability(modelID)["image_generation"]?.object
    else { throw ConversationFailure.stale }
    let row = try model(modelID)
    guard let port = row["port"]?.integer, let port = UInt16(exactly: port) else {
      throw ConversationFailure.invalid("Choose a running image model")
    }
    let params = NativeImageGeneration.defaults.merging(
      document.fields["imageParams"]?.object ?? [:]
    ) { _, new in new }
    let plan = try NativeImageGeneration.plan(document: document, message: message, caps: cap)
    let seed = NativeImageGeneration.seed(
      params, document: document, message: message, editing: !plan.references.isEmpty
    ) {
      Int.random(in: 0...Int(Int32.max))
    }
    let prompt = plan.prompt
    let body = try NativeImageGeneration.body(
      model: modelID, prompt: prompt, params: params, seed: seed, caps: cap)
    let steps =
      params["steps"]?.integer
      ?? (params["quality"]?.string == "low"
        ? 20 : params["quality"]?.string == "medium" ? 30 : cap["default_steps"]?.integer ?? 40)
    let size =
      body["size"] ?? (plan.references.isEmpty ? cap["default_size"] : nil) ?? .string("auto")
    try updateMessage(messageID) {
      $0["content"] = .array([])
      $0["run"] = .object([
        "model": .string(modelID), "modelName": row["title"] ?? .string(modelID),
        "vendor": row["vendor"] ?? .string(""), "tools": .array([]), "at": Self.now,
      ])
      $0["imageGen"] = .object([
        "prompt": .string(prompt), "params": .object(params),
        "seed": .number(Decimal(seed)), "steps": .number(Decimal(steps)), "size": size,
        "previews": .number(0),
        "references": plan.references.isEmpty ? .null : .number(Decimal(plan.references.count)),
        "referencesFrom": plan.from.map(V.string) ?? .null,
      ])
    }
    schedulePublish()
    let started = ContinuousClock.now
    let reply = try await transport.images(port: port, body: body, references: plan.references) {
      [weak self] event in
      try await self?.receiveImagePreview(event, messageID: messageID)
    }
    let seconds = Self.seconds(started.duration(to: .now))
    try Task.checkCancellation()
    let format = reply["output_format"]?.string ?? params["format"]?.string ?? "png"
    guard ["png", "jpeg", "webp"].contains(format) else {
      throw ConversationFailure.invalid("Unknown image format")
    }
    let images = reply["data"]?.array ?? []
    guard images.count == params["n"]?.integer ?? 1 else {
      throw ConversationFailure.invalid("The image endpoint returned an incomplete result")
    }
    for (index, item) in images.enumerated() {
      guard let encoded = item["b64_json"]?.string, let bytes = Data(base64Encoded: encoded),
        !bytes.isEmpty, bytes.count <= 48 * 1024 * 1024
      else {
        throw ConversationFailure.invalid("The endpoint returned an invalid picture")
      }
      let id = UUID().uuidString
      let name =
        "\(reply["size"]?.string ?? size.string ?? "image")-seed\(seed)\(images.count > 1 ? "-\(index + 1)" : "").\(format)"
      var part: O = [
        "type": .string("image"), "attachmentId": .string(id), "mime": .string("image/\(format)"),
        "name": .string(name), "size": .number(Decimal(bytes.count)),
        "gen": .object([
          "seed": .number(Decimal(seed)), "steps": .number(Decimal(steps)),
          "size": reply["size"] ?? size,
          "quality": reply["quality"] ?? params["quality"]!, "format": .string(format),
          "background": reply["background"] ?? params["background"]!,
        ]),
      ]
      do {
        _ = try await transport.bytes(
          "api/attachments/\(id)", method: "PUT", body: bytes,
          contentType: "image/\(format)", query: ["name": name, "conv": document.id])
      } catch {
        // Match web Studio: a completed render must not disappear because
        // the attachment store failed. Keep the original bytes, not a thumbnail.
        part["attachmentId"] = .string("")
        part["dataUrl"] = .string("data:image/\(format);base64,\(encoded)")
      }
      try updateMessage(messageID) {
        $0["content"] = .array(($0["content"]?.array ?? []) + [.object(part)])
      }
      // Checkpoint each completed picture; a later failure never discards one.
      try await persist()
    }
    try updateMessage(messageID) {
      var recipe = $0["imageGen"]?.object ?? [:]
      recipe["elapsedS"] = .number(Decimal(seconds))
      recipe["size"] = reply["size"] ?? size
      recipe["sPerStep"] = .number(
        Decimal(
          NativeImageGeneration.secondsPerStep(
            elapsed: seconds, steps: steps, images: images.count)))
      $0["imageGen"] = .object(recipe)
      $0["usage"] = .object([
        "promptTokens": reply["usage"]?["input_tokens"] ?? .number(0),
        "completionTokens": reply["usage"]?["output_tokens"] ?? .number(0),
        "ttftMs": reply["firstPreviewMs"] ?? .null,
        "ms": .number(Decimal(seconds * 1000)),
      ])
    }
  }
  func receiveImagePreview(_ event: O, messageID: String) throws {
    try Task.checkCancellation()
    guard tasks[messageID] != nil, let b64 = event["b64_json"]?.string,
      b64.utf8.count <= 64 * 1024 * 1024
    else { throw ConversationFailure.stale }
    let format = event["output_format"]?.string ?? "png"
    guard ["png", "jpeg", "webp"].contains(format) else {
      throw ConversationFailure.invalid("Unknown preview format")
    }
    let index = event["partial_image_index"]?.integer ?? 0
    imagePreviews[messageID] = [
      "id": .string("\(messageID)-preview-\(index)"),
      "name": .string("Preview \(index + 1)"), "preview": .bool(true),
      "dataURL": .string("data:image/\(format);base64,\(b64)"),
    ]
    try updateMessage(messageID) {
      var recipe = $0["imageGen"]?.object ?? [:]
      recipe["previews"] = .number(Decimal(index + 1))
      $0["imageGen"] = .object(recipe)
    }
    schedulePublish()
  }
}
