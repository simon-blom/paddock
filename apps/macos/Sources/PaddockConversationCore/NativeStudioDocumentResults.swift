import Foundation

extension NativeStudioRuntime {
  func documentResult(_ message: O) -> V {
    guard message["ocr"] != nil || message["docRun"] != nil else { return .null }
    let facts = NativeOCRMetadata.facts(message["ocr"])
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
            "number": page["page"] ?? .number(Decimal(index + 1)),
            "pdfPage": page["page"] ?? .null,
            "sourceID": message["docRun"]?["sourceId"] ?? .null,
            "attachmentID": page["attachmentId"] ?? .null,
            "name": page["name"] ?? .null,
            "text": .string(
              documentDisplay.display(
                id: (message["id"]?.string ?? "") + "/\(index)", raw: page["text"]?.string ?? "")),
            "note": page["note"] ?? .string(""),
            "regions": .array(
              NativeDocumentRunState.regions(page["regions"]?.array ?? []).map {
                .object([
                  "label": $0["label"] ?? .string(""), "text": $0["text"] ?? .string(""),
                  "boxes": $0["boxes"] ?? .array([]), "quads": $0["quads"] ?? .array([]),
                ])
              }),
            "unsure": .array(Self.unsureDocumentWords(page)),
          ])
        }),
    ])
  }

  private static func unsureDocumentWords(_ page: V) -> [V] {
    guard !["reading", "queued"].contains(page["state"]?.string ?? "") else { return [] }
    var seen = Set<String>()
    return (page["words"]?.array ?? []).prefix(65536).compactMap { word in
      guard let confidence = word["c"]?.double, confidence.isFinite, confidence >= 0,
        confidence < 0.45, let text = word["w"]?.string, text.count > 1,
        text.utf8.count <= 256, seen.insert(text).inserted
      else { return nil }
      return .object([
        "label": .string(text), "value": .string("\(Int((confidence * 100).rounded()))%"),
      ])
    }
  }
}
