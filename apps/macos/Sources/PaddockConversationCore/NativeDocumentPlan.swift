import Foundation

/// A retained native PDF document, not an array of rasterized pages. The host
/// owns CoreGraphics; the conversation actor only asks for the next page.
public struct NativeDocumentSource: Sendable {
  public let pageCount: Int
  public let render: @Sendable (Int, Int) async throws -> NativeRasterPage
  public init(
    pageCount: Int, render: @escaping @Sendable (Int, Int) async throws -> NativeRasterPage
  ) {
    self.pageCount = pageCount
    self.render = render
  }
}

public struct NativeRasterPage: Sendable {
  public let image: [String: ConversationValue]
  public let width: Int
  public let height: Int
  public init(image: [String: ConversationValue], width: Int, height: Int) {
    self.image = image
    self.width = width
    self.height = height
  }
}

struct NativeDocumentPlan: Sendable {
  typealias V = ConversationValue
  typealias O = [String: V]
  let sourceID: String
  let parts: [O]
  let instruction: String

  static func isPDF(_ part: O) -> Bool {
    part["type"]?.string == "file"
      && ((part["mime"]?.string ?? "").contains("pdf")
        || (part["name"]?.string ?? "").lowercased().hasSuffix(".pdf"))
      && part["pdfMode"]?.string != "text"
  }
  static func rasterParts(_ message: O) -> [O] {
    (message["content"]?.array ?? []).compactMap(\.object).filter {
      $0["type"]?.string == "image" || isPDF($0)
    }
  }
  static func make(document: ConversationDocument, capability: O) -> Self? {
    let parser = capability["document_parser"]?.bool == true
    let tags = capability["task_tags"]?.array ?? []
    guard parser || !tags.isEmpty else { return nil }
    let path = document.activeMessages
    guard let user = path.last(where: { $0["role"]?.string == "user" }) else { return nil }
    // A newly attached DOCX/text-only PDF belongs to this turn, not an older
    // raster document that happens to remain selected in the viewer.
    if rasterParts(user).isEmpty,
      (user["content"]?.array ?? []).contains(where: { $0["type"]?.string == "file" })
    {
      return nil
    }
    let documents = path.filter { $0["role"]?.string == "user" && !rasterParts($0).isEmpty }
    // Fresh raster input wins. Otherwise the selected document is sticky,
    // restricted to the branch on screen; never resurrect a sibling's file.
    let source =
      !rasterParts(user).isEmpty
      ? user
      : documents.first { $0["id"] == document.fields["activeDocId"] } ?? documents.last
    guard let source, let sourceID = source["id"]?.string else { return nil }
    let parts = rasterParts(source)
    let instruction = ConversationDocument.text(user).trimmingCharacters(
      in: .whitespacesAndNewlines)
    if !parser {
      guard NativeDocumentRequestOptions.isTask(instruction, capability: capability),
        parts.contains(where: isPDF) || parts.count > 1
      else { return nil }
    }
    return Self(sourceID: sourceID, parts: parts, instruction: instruction)
  }

  static func pages(_ range: String?, count: Int, limit: Int) throws -> ClosedRange<Int> {
    guard count > 0, count <= 1_000_000 else {
      throw ConversationFailure.invalid("Invalid PDF page count")
    }
    var first = 1
    var last = count
    if let range, !range.isEmpty {
      let pieces = range.split(separator: "-", omittingEmptySubsequences: false)
      guard pieces.count <= 2, let start = pieces.first.flatMap({ Int($0) }), start >= 1 else {
        throw ConversationFailure.invalid("Invalid PDF page range")
      }
      first = start
      if pieces.count == 1 {
        last = start
      } else if !pieces[1].isEmpty {
        guard let end = Int(pieces[1]) else {
          throw ConversationFailure.invalid("Invalid PDF page range")
        }
        last = end
      }
    }
    guard first <= last, last <= count else {
      throw ConversationFailure.invalid("The PDF page range is outside the document")
    }
    guard last - first < limit else {
      throw ConversationFailure.invalid("Select up to \(limit) PDF pages per turn")
    }
    return first...last
  }
}
