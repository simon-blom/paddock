import Foundation

/// Timestamp enrichment never changes the recognized words. Compare normalized
/// Unicode sequences (including CJK one-character aligner tokens), reject any
/// mismatch or invalid clock, and preserve confidence/unknown metadata fields.
public enum NativeSpeechAlignment {
  public typealias V = ConversationValue
  public typealias O = [String: V]
  static func fold(_ text: String) -> String {
    let excluded = CharacterSet.punctuationCharacters.union(.symbols).union(.whitespacesAndNewlines)
    return String(
      String.UnicodeScalarView(
        text.lowercased().precomposedStringWithCompatibilityMapping.unicodeScalars.filter {
          !excluded.contains($0)
        }))
  }
  public static func merge(meta: O, text: String, aligned: [V], duration: Double?) -> [V]? {
    guard !aligned.isEmpty else { return nil }
    var previous = 0.0
    for word in aligned {
      guard let start = word["start"]?.double, let end = word["end"]?.double,
        start.isFinite, end.isFinite, start >= previous, end >= start,
        duration.map({ end <= $0 + 0.001 }) ?? true
      else { return nil }
      previous = end
    }
    let old = meta["words"]?.array ?? []
    let words =
      old.isEmpty
      ? text.split(whereSeparator: \.isWhitespace).map { V.object(["word": .string(String($0))]) }
      : old
    guard !words.isEmpty else { return nil }
    var cursor = 0
    var result: [V] = []
    for value in words {
      var word = value.object ?? [:]
      let target = fold(word["word"]?.string ?? "")
      if target.isEmpty {
        result.append(.object(word))
        continue
      }
      var matched = ""
      var start: V?
      var end: V?
      while matched.count < target.count {
        guard cursor < aligned.count else { return nil }
        let source = aligned[cursor]
        cursor += 1
        let part = fold(source["word"]?.string ?? "")
        if part.isEmpty { continue }
        matched += part
        guard target.hasPrefix(matched) else { return nil }
        start = start ?? source["start"]
        end = source["end"]
      }
      guard matched == target else { return nil }
      word["start"] = start
      word["end"] = end
      result.append(.object(word))
    }
    guard aligned.dropFirst(cursor).allSatisfy({ fold($0["word"]?.string ?? "").isEmpty }) else {
      return nil
    }
    return result
  }
}

extension NativeStudioRuntime {
  func enrichSpeech(messageID: String, clip: O) async throws {
    let conversationID = document?.id
    guard let message = document?.messages.first(where: { $0["id"]?.string == messageID }),
      let meta = message["transcript"]?.object,
      !(meta["words"]?.array ?? []).contains(where: {
        $0["start"]?.double != nil && $0["end"]?.double != nil
      }),
      let lane = models.first(where: {
        $0["kind"]?.string == "aligner" && $0["status"]?.string == "ok"
          && $0["port"]?.integer != nil
      }),
      let id = lane["id"]?.string, let port = lane["port"]?.integer
    else { return }
    let text = ConversationDocument.text(message)
    let language = meta["language"]?.string ?? clip["language"]?.string
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !["ja", "japanese", "ko", "korean"].contains(language?.lowercased() ?? "")
    else { return }
    let duration = meta["durationS"]?.double ?? clip["durationS"]?.double
    if let limit = capability(id)["alignment_max_clip_s"]?.double, let duration, duration > limit {
      return
    }
    do {
      let aid = try Self.id(clip["attachmentId"])
      let bytes = try await transport.bytes("api/attachments/\(aid)", maximum: 100 * 1024 * 1024)
      let boundary = UUID().uuidString
      var body = Data()
      for (key, value) in [("text", text), ("language", language == "auto" ? nil : language)] {
        if let value {
          body.append(
            Data(
              "--\(boundary)\r\nContent-Disposition: form-data; name=\"\(key)\"\r\n\r\n\(value)\r\n"
                .utf8))
        }
      }
      let mime = clip["mime"]?.string ?? "audio/wav"
      guard !mime.contains("\r"), !mime.contains("\n") else {
        throw ConversationFailure.invalid("Invalid audio type")
      }
      body.append(
        Data(
          "--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording\"\r\nContent-Type: \(mime)\r\n\r\n"
            .utf8))
      body.append(bytes)
      body.append(Data("\r\n--\(boundary)--\r\n".utf8))
      let data = try await transport.bytes(
        "api/runners/\(port)/v1/audio/alignments", method: "POST", body: body,
        contentType: "multipart/form-data; boundary=\(boundary)")
      let response = try JSONDecoder().decode(O.self, from: data)
      guard
        let words = NativeSpeechAlignment.merge(
          meta: meta, text: text, aligned: response["words"]?.array ?? [], duration: duration)
      else {
        throw ConversationFailure.invalid(
          "Alignment did not match the transcript; original text retained")
      }
      guard document?.id == conversationID,
        document?.messages.first(where: { $0["id"]?.string == messageID }).map(
          ConversationDocument.text) == text
      else { return }
      try updateMessage(messageID) { row in
        var enriched = row["transcript"]?.object ?? meta
        enriched["words"] = .array(words)
        enriched["wordsFrom"] = .string(id)
        enriched["wordsLangOk"] = response["language_supported"] ?? .bool(true)
        row["transcript"] = .object(enriched)
      }
    } catch {
      try Task.checkCancellation()
      guard document?.id == conversationID else { return }
      // Optional enrichment cannot turn a successful transcription into a
      // failed turn, but the unavailable times remain explicit in its metadata.
      try updateMessage(messageID) { row in
        var value = row["transcript"]?.object ?? meta
        value["alignmentError"] = .string(error.localizedDescription)
        row["transcript"] = .object(value)
      }
    }
  }
}
