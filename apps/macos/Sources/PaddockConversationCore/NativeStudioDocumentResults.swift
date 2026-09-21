import Foundation

extension NativeStudioRuntime {
  func documentResult(_ message: O) -> V {
    guard message["ocr"] != nil || message["docRun"] != nil else { return .null }
    func scalar(_ value: V) -> String? {
      switch value {
      case .string(let text): text
      case .number(let n): NSDecimalNumber(decimal: n).stringValue
      case .bool(let b): b ? "true" : "false"
      default: nil
      }
    }
    let facts = (message["ocr"]?.object ?? [:]).sorted { $0.key < $1.key }.compactMap {
      key, value -> V? in
      guard let value = scalar(value) else { return nil }
      return .object(["label": .string(key), "value": .string(value)])
    }
    let pages =
      message["docRun"]?["pages"]?.array ?? [
        .object([
          "state": .string(message["streaming"]?.bool == true ? "reading" : "done"),
          "text": .string(ConversationDocument.text(message)),
          "regions": message["ocr"]?["regions"] ?? .array([]),
        ])
      ]
    return .object([
      "facts": .array(facts),
      "pages": .array(
        pages.enumerated().map { index, page in
          .object([
            "id": .number(Decimal(index + 1)), "state": page["state"] ?? .string("done"),
            "text": page["text"] ?? .string(""), "note": page["note"] ?? .string(""),
            "regions": .array(
              (page["regions"]?.array ?? []).map {
                .object([
                  "label": $0["label"] ?? .string(""), "text": $0["text"] ?? .string(""),
                  "boxes": $0["boxes"] ?? .array([]), "quads": $0["quads"] ?? .array([]),
                ])
              }),
            "unsure": .array(
              (page["words"]?.array ?? []).compactMap { word in
                guard let confidence = word["c"]?.double, confidence < 0.45 else { return nil }
                return .object([
                  "label": word["w"] ?? .string(""),
                  "value": .string("\(Int((confidence * 100).rounded()))%"),
                ])
              }),
          ])
        }),
    ])
  }
}
