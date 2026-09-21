import Foundation

/// Small, explicit language policies. Unknown/plain text never inherits C-style
/// comments or quote insertion merely because it happens to contain punctuation.
enum ArtifactEditing {
  struct Edit {
    let range: NSRange
    let text: String
  }
  enum Comment: Equatable {
    case line(String)
    case block(String, String)
  }
  static func comment(for language: String) -> Comment? {
    switch ArtifactSourceSyntax.normalize(language) {
    case "html", "xml", "markdown": .block("<!--", "-->")
    case "css", "scss", "less": .block("/*", "*/")
    case "python", "bash", "yaml", "ruby", "toml", "r", "dockerfile": .line("#")
    case "sql", "lua": .line("--")
    case "mermaid": .line("%%")
    case "javascript", "typescript", "swift", "rust", "c", "cpp", "csharp", "java", "kotlin", "go",
      "cypher", "jsonc":
      .line("//")
    default: nil  // JSON deliberately has no comment syntax.
    }
  }
  static func allowsPairs(_ language: String) -> Bool {
    !["", "text", "plaintext", "csv", "markdown", "mermaid"].contains(
      ArtifactSourceSyntax.normalize(language))
      && (comment(for: language) != nil || language == "json")
  }
  static let pairs: [UInt16: UInt16] = [40: 41, 91: 93, 123: 125, 34: 34, 39: 39, 96: 96]

  static func lineRanges(_ text: NSString, selections: [NSRange]) -> [NSRange] {
    var starts = Set<Int>()
    var result: [NSRange] = []
    for selection in selections {
      guard selection.location <= text.length, NSMaxRange(selection) <= text.length else {
        continue
      }
      let end = selection.length > 0 ? NSMaxRange(selection) - 1 : selection.location
      var position = text.lineRange(for: NSRange(location: selection.location, length: 0)).location
      repeat {
        let range = text.lineRange(for: NSRange(location: position, length: 0))
        if starts.insert(position).inserted { result.append(range) }
        let next = NSMaxRange(range)
        if next <= position || next > end { break }
        position = next
      } while position <= text.length
    }
    return result.sorted { $0.location < $1.location }
  }

  static func comments(_ text: NSString, selections: [NSRange], style: Comment) -> [Edit] {
    switch style {
    case .line(let marker):
      let rows = lineRanges(text, selections: selections).compactMap { range -> (Int, String)? in
        let row = text.substring(with: range).trimmingCharacters(in: .newlines)
        let indent = row.prefix { $0 == " " || $0 == "\t" }.utf16.count
        guard indent < row.utf16.count else { return nil }
        return (range.location + indent, String(row.dropFirst(indent)))
      }
      let remove = !rows.isEmpty && rows.allSatisfy { $0.1.hasPrefix(marker) }
      return rows.map { offset, value in
        Edit(
          range: NSRange(
            location: offset,
            length: remove
              ? marker.utf16.count + (value.dropFirst(marker.count).first == " " ? 1 : 0) : 0),
          text: remove ? "" : marker + " ")
      }
    case .block(let open, let close):
      guard selections.count == 1 else { return [] }
      var range = selections[0]
      if range.length == 0 {
        range = text.lineRange(for: range)
        let row = text.substring(with: range)
        let leading = row.prefix { $0 == " " || $0 == "\t" }.utf16.count
        let ending = row.reversed().prefix { $0 == "\r" || $0 == "\n" || $0 == "\r\n" }.reduce(0) {
          $0 + String($1).utf16.count
        }
        range = NSRange(
          location: range.location + leading, length: max(0, range.length - leading - ending))
      }
      let value = text.substring(with: range)
      if value.hasPrefix(open), value.hasSuffix(close),
        value.utf16.count >= open.utf16.count + close.utf16.count
      {
        var inner = String(value.dropFirst(open.count).dropLast(close.count))
        if inner.hasPrefix(" ") { inner.removeFirst() }
        if inner.hasSuffix(" ") { inner.removeLast() }
        return [Edit(range: range, text: inner)]
      }
      guard !value.contains(open), !value.contains(close) else { return [] }
      return [Edit(range: range, text: open + " " + value + " " + close)]
    }
  }

  static func translated(_ offset: Int, through edits: [Edit]) -> Int {
    var result = offset
    for edit in edits where edit.range.location <= offset {
      result += edit.text.utf16.count - min(edit.range.length, offset - edit.range.location)
    }
    return result
  }
}
