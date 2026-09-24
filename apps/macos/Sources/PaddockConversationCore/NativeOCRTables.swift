import Foundation
import libxml2

/// Inert, bounded HTML-to-text conversion using macOS's HTML parser. Never use
/// NSAttributedString's HTML importer here: it can instantiate a web renderer.
/// The raw response remains saved; incomplete/oversized tables stay untouched.
enum NativeOCRTables {
  private static let closed = try! NSRegularExpression(
    pattern: #"<table\b[\s\S]*?</table\s*>"#, options: .caseInsensitive)

  static func display(_ text: String) -> String {
    guard text.utf8.count <= 4 * 1024 * 1024,
      text.range(of: "<table", options: .caseInsensitive) != nil
    else { return text }
    let source = text as NSString
    let result = NSMutableString(string: text)
    for match in closed.matches(in: text, range: NSRange(location: 0, length: source.length))
      .reversed()
    {
      let raw = source.substring(with: match.range)
      if let table = convert(raw) { result.replaceCharacters(in: match.range, with: table) }
    }
    return result as String
  }

  private static func convert(_ raw: String) -> String? {
    guard raw.utf8.count <= 1024 * 1024 else { return nil }
    let options =
      HTML_PARSE_NONET.rawValue | HTML_PARSE_NOERROR.rawValue
      | HTML_PARSE_NOWARNING.rawValue | HTML_PARSE_NODEFDTD.rawValue
    guard
      let doc = raw.withCString({
        htmlReadMemory($0, Int32(raw.utf8.count), nil, "UTF-8", Int32(options))
      })
    else { return nil }
    defer { xmlFreeDoc(doc) }
    var rows: [[String]] = []
    var cells = 0
    var nodes = 0
    var tables = 0
    func name(_ node: xmlNodePtr) -> String { String(cString: node.pointee.name) }
    func walk(_ start: xmlNodePtr?, depth: Int) -> Bool {
      guard depth <= 64 else { return false }
      var next = start
      while let node = next {
        nodes += 1
        guard nodes <= 16384 else { return false }
        next = node.pointee.next
        let tag = name(node)
        if tag == "table" {
          tables += 1
          guard tables == 1 else { return false }  // No duplicate nested-cell extraction.
        }
        if tag == "tr" {
          var row: [String] = []
          var child = node.pointee.children
          while let cell = child {
            child = cell.pointee.next
            guard ["td", "th"].contains(name(cell)) else { continue }
            cells += 1
            guard cells <= 4096, row.count < 256 else { return false }
            guard let buffer = xmlBufferCreate() else { return false }
            let status = xmlNodeBufGetContent(buffer, cell)
            let value = xmlBufferContent(buffer).map { String(cString: $0) } ?? ""
            xmlBufferFree(buffer)
            guard status == 0 else { return false }
            let folded = value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            // Cell text is data, not model-authored Markdown or executable HTML.
            let escaped = folded.reduce(into: "") { out, character in
              if #"\`*_{}[]<>|!"#.contains(character) { out.append("\\") }
              out.append(character)
            }
            row.append(escaped)
          }
          if !row.isEmpty { rows.append(row) }
        }
        if !walk(node.pointee.children, depth: depth + 1) { return false }
      }
      return true
    }
    guard walk(xmlDocGetRootElement(doc), depth: 0), tables == 1,
      let width = rows.map(\.count).max(), width > 0
    else { return nil }
    func line(_ row: [String]) -> String {
      "| " + (row + Array(repeating: "", count: width - row.count)).joined(separator: " | ") + " |"
    }
    let lines =
      [line(rows[0]), "|" + String(repeating: " --- |", count: width)]
      + rows.dropFirst().map(line)
    return "\n\n" + lines.joined(separator: "\n") + "\n\n"
  }
}
