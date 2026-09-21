import Foundation

/// Original, deliberately lexical (not a language server). UTF-16 coordinates
/// match AppKit, including emoji, combining marks and CRLF. Cached line input
/// and output states let an edit converge without re-tokenizing its suffix.
struct ArtifactSourceSyntax: Sendable {
  enum Kind: Sendable { case keyword, string, comment, number, tag, attribute, bracket }
  struct Token: Sendable {
    let range: NSRange  // relative to the logical line
    let kind: Kind
  }
  struct State: Equatable, Sendable {
    var blockEnd = ""
    var quote: UInt16 = 0
    var tripleQuote = false
    var tag = false
    var tagName = ""
    var closingTag = false
    var embedded = ""
    var fence = ""
    var lineComment = false
  }
  struct Line: Sendable {
    let text: String
    let input: State
    let output: State
    let tokens: [Token]
    let words: [String]
  }
  private struct Lexed {
    let tokens: [Token]
    let words: [String]
  }
  let language: String
  let starts: [Int]
  let lines: [Line]
  let length: Int
  let lexedLines: Int
  let brackets: [Int: Int]
  let words: [String]

  func line(at offset: Int) -> Int {
    var lo = 0
    var hi = starts.count
    while lo < hi {
      let mid = (lo + hi) / 2
      if starts[mid] <= offset { lo = mid + 1 } else { hi = mid }
    }
    return max(0, lo - 1)
  }

  static func normalize(_ language: String) -> String {
    switch language.lowercased() {
    case "js", "jsx", "javascript": "javascript"
    case "ts", "tsx", "typescript": "typescript"
    case "svg", "xml": "xml"
    case "htm", "html": "html"
    case "py", "python": "python"
    case "sh", "zsh", "shell", "bash": "bash"
    case "yml": "yaml"
    case "md": "markdown"
    case "c++", "cc", "cxx", "h", "hpp": "cpp"
    case "c#", "cs": "csharp"
    case "rs": "rust"
    default: language.lowercased()
    }
  }

  static func parse(_ text: String, language: String, previous: Self? = nil) throws -> Self {
    let language = normalize(language)
    let old = previous?.language == language ? previous : nil
    let ns = text as NSString
    var starts: [Int] = []
    var strings: [String] = []
    var offset = 0
    // getLineStart handles CR, LF, CRLF and Unicode paragraph separators.
    while offset < ns.length {
      try Task.checkCancellation()
      let range = ns.lineRange(for: NSRange(location: offset, length: 0))
      starts.append(offset)
      strings.append(ns.substring(with: range))
      offset = NSMaxRange(range)
    }
    if ns.length == 0 || [10, 13, 0x2028, 0x2029].contains(ns.character(at: ns.length - 1)) {
      starts.append(ns.length)
      strings.append("")
    }
    var prefix = 0
    var suffix = 0
    if let old {
      while prefix < min(strings.count, old.lines.count), strings[prefix] == old.lines[prefix].text
      { prefix += 1 }
      while suffix < min(strings.count, old.lines.count) - prefix,
        strings[strings.count - suffix - 1] == old.lines[old.lines.count - suffix - 1].text
      { suffix += 1 }
    }
    var result: [Line] = []
    var state = State()
    var lexed = 0
    result.reserveCapacity(strings.count)
    for i in strings.indices {
      try Task.checkCancellation()
      let oldIndex =
        i < prefix
        ? i : (i >= strings.count - suffix ? i + (old?.lines.count ?? 0) - strings.count : -1)
      if let old, oldIndex >= 0, old.lines[oldIndex].input == state {
        let cached = old.lines[oldIndex]
        result.append(cached)
        state = cached.output
      } else {
        let input = state
        let value = try lex(strings[i], language: language, state: &state)
        result.append(
          Line(
            text: strings[i], input: input, output: state, tokens: value.tokens, words: value.words)
        )
        lexed += 1
      }
    }
    var brackets: [Int: Int] = [:]
    var stack: [(UInt16, Int)] = []
    var words = Set<String>()
    for (index, line) in result.enumerated() {
      try Task.checkCancellation()
      if line.tokens.contains(where: { $0.kind == .bracket }) {
        let units = Array(line.text.utf16)
        for token in line.tokens where token.kind == .bracket {
          let character = units[token.range.location]
          let position = starts[index] + token.range.location
          if let closing = ArtifactEditing.pairs[character] {
            stack.append((closing, position))
          } else if let opening = stack.last, opening.0 == character {
            stack.removeLast()
            brackets[opening.1] = position
            brackets[position] = opening.1
          } else {
            stack.removeAll(keepingCapacity: true)
          }
        }
      }
      // Words come from the existing lexer pass, and travel with cached lines.
      // Never rescan the document (or retain string/comment bodies) for completion.
      for word in line.words where words.count < 8192 { words.insert(word) }
    }
    return Self(
      language: language, starts: starts, lines: result, length: ns.length, lexedLines: lexed,
      brackets: brackets, words: words.sorted())
  }

  private static func identifierStart(_ c: UInt16) -> Bool {
    (65...90).contains(c) || (97...122).contains(c) || c == 95 || c == 36
  }

  struct Context {
    let language: String
    let state: State
    var isLiteral: Bool { state.quote != 0 || !state.blockEnd.isEmpty || state.lineComment }
  }
  /// Start from cached state before the earliest unparsed edit. This keeps rapid
  /// typing near EOF local while still propagating an earlier comment/string
  /// change. An excessively long unparsed span declines assistance, not input.
  static func context(
    _ text: String, offset: Int, language: String, current: Self?, unchangedThrough: Int = .max
  ) -> Context? {
    let ns = text as NSString
    guard offset >= 0, offset <= ns.length else { return nil }
    var state = State()
    var start = 0
    if let current {
      let index = current.line(at: min(offset, unchangedThrough))
      state = current.lines[index].input
      start = current.starts[index]
    }
    guard offset - start <= 32768 else { return nil }
    let prefix = ns.substring(with: NSRange(location: start, length: offset - start)) as NSString
    var position = 0
    while position < prefix.length {
      let range = prefix.lineRange(for: NSRange(location: position, length: 0))
      guard
        (try? lex(prefix.substring(with: range), language: normalize(language), state: &state))
          != nil
      else { return nil }
      position = NSMaxRange(range)
    }
    if offset > 0, [10, 13].contains(ns.character(at: offset - 1)) { state.lineComment = false }
    return Context(
      language: state.embedded.isEmpty ? normalize(language) : state.embedded, state: state)
  }

  private static let keywords: Set<String> = Set(
    ("as async await break case catch class const continue debugger default defer delete do else enum export extends false finally for from func function guard if implements import in init instanceof interface is let match new nil null of override private public repeat return self static struct super switch this throw throws true try type typeof var void while yield undefined actor some "
      + "and assert def elif except False global lambda None nonlocal not or pass raise True with "
      + "SELECT MATCH WHERE RETURN CREATE DELETE SET WITH UNWIND ORDER BY LIMIT OPTIONAL MERGE "
      + "graph flowchart sequenceDiagram classDiagram stateDiagram subgraph end participant LR TD TB BT RL "
      + "fn impl trait pub mut use mod crate move dyn where unsafe extern ref inout extension protocol associatedtype typealias fileprivate internal open weak unowned final nonisolated isolated "
      + "auto bool char double float int long short signed unsigned size_t sizeof typedef union virtual volatile namespace template typename using constexpr noexcept nullptr protected package synchronized synchronized void boolean byte string val fun object data sealed suspend when range map chan select go make defer")
      .split(separator: " ").map(String.init))

  private static func lex(_ text: String, language: String, state: inout State) throws -> Lexed {
    state.lineComment = false
    guard !["", "text", "plaintext", "csv"].contains(language) else {
      return Lexed(tokens: [], words: [])
    }
    let u = Array(text.utf16)
    var i = 0
    var tokens: [Token] = []
    var words = Set<String>()
    func remember(_ word: String) {
      if words.count < 512, (2...64).contains(word.utf16.count) { words.insert(word) }
    }
    let markup = language == "html" || language == "xml"
    if language == "markdown" || language == "md" {
      let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
      if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
        let marker = String(trimmed.prefix { $0 == trimmed.first })
        if state.fence.isEmpty {
          state.fence = marker
          state.embedded = normalize(
            String(trimmed.dropFirst(marker.count)).trimmingCharacters(in: .whitespaces))
          if state.embedded.isEmpty { state.embedded = "text" }
        } else if marker.first == state.fence.first && marker.count >= state.fence.count {
          state = State()
        }
        return Lexed(
          tokens: [Token(range: NSRange(location: 0, length: u.count), kind: .keyword)], words: [])
      }
      if state.fence.isEmpty {
        if trimmed.hasPrefix("#") || trimmed.hasPrefix(">") {
          return Lexed(
            tokens: [Token(range: NSRange(location: 0, length: u.count), kind: .keyword)], words: []
          )
        }
        return Lexed(tokens: [], words: [])
      }
      if state.embedded == "text" { return Lexed(tokens: [], words: []) }
    }
    func has(_ value: String, at position: Int) -> Bool {
      let chars = Array(value.utf16)
      return position + chars.count <= u.count
        && u[position..<position + chars.count].elementsEqual(chars)
    }
    func word(_ c: UInt16) -> Bool {
      (65...90).contains(c) || (97...122).contains(c) || (48...57).contains(c) || c == 95 || c == 36
        || c >= 128
    }
    func rawTextEnd(at position: Int) -> Bool {
      guard markup, !state.embedded.isEmpty, position < u.count, u[position] == 60 else {
        return false
      }
      let close = Array((state.embedded == "css" ? "</style" : "</script").utf16)
      guard position + close.count <= u.count else { return false }
      for j in close.indices {
        let c = u[position + j]
        if ((65...90).contains(c) ? c + 32 : c) != close[j] { return false }
      }
      let end = position + close.count
      return end == u.count || [9, 10, 13, 32, 47, 62].contains(u[end])
    }
    func emit(_ start: Int, _ kind: Kind) {
      if i > start {
        tokens.append(Token(range: NSRange(location: start, length: i - start), kind: kind))
      }
    }
    while i < u.count {
      if i % 1024 == 0 { try Task.checkCancellation() }
      let start = i
      // HTML raw-text end tags terminate script/style even inside JS strings.
      if rawTextEnd(at: i) {
        state.embedded = ""
        state.quote = 0
        state.blockEnd = ""
        state.lineComment = false
      }
      if !state.blockEnd.isEmpty {
        while i < u.count, !has(state.blockEnd, at: i), !rawTextEnd(at: i) {
          if i % 4096 == 0 { try Task.checkCancellation() }
          i += 1
        }
        if rawTextEnd(at: i) {
          emit(start, .comment)
          continue
        }
        if i < u.count {
          i += state.blockEnd.utf16.count
          state.blockEnd = ""
        }
        emit(start, .comment)
        continue
      }
      if state.quote != 0 {
        while i < u.count {
          if i % 4096 == 0 { try Task.checkCancellation() }
          if rawTextEnd(at: i) { break }
          if u[i] == 92, !state.tag {
            i = min(u.count, i + 2)
            continue
          }
          if state.tripleQuote {
            if i + 2 < u.count, u[i] == state.quote, u[i + 1] == state.quote,
              u[i + 2] == state.quote
            {
              i += 3
              state.quote = 0
              state.tripleQuote = false
              break
            }
            i += 1
            continue
          }
          let c = u[i]
          i += 1
          if c == state.quote {
            state.quote = 0
            break
          }
        }
        emit(start, .string)
        continue
      }
      if markup, state.embedded.isEmpty {
        if has("<!--", at: i) {
          state.blockEnd = "-->"
          i += 4
          emit(start, .comment)
          continue
        }
        if u[i] == 60 {
          state.tag = true
          state.closingTag = has("</", at: i)
          state.tagName = ""
          i += state.closingTag ? 2 : 1
          emit(start, .tag)
          continue
        }
        if state.tag {
          if u[i] == 62 {
            state.tag = false
            if !state.closingTag, language == "html" {
              if state.tagName == "script" { state.embedded = "javascript" }
              if state.tagName == "style" { state.embedded = "css" }
            }
            i += 1
            emit(start, .tag)
            continue
          }
          if u[i] == 34 || u[i] == 39 {
            state.quote = u[i]
            i += 1
            emit(start, .string)
            continue
          }
          if word(u[i]) || u[i] == 45 {
            while i < u.count, word(u[i]) || u[i] == 45 || u[i] == 58 { i += 1 }
            let kind: Kind = state.tagName.isEmpty ? .tag : .attribute
            let name = String(decoding: u[start..<i], as: UTF16.self)
            remember(name)
            if state.tagName.isEmpty { state.tagName = name.lowercased() }
            emit(start, kind)
            continue
          }
        }
        i += 1
        continue
      }
      let lang = state.embedded.isEmpty ? language : state.embedded
      let hashComments = ["python", "bash", "yaml", "ruby", "toml"].contains(lang)
      if case .line(let marker) = ArtifactEditing.comment(for: lang), has(marker, at: i) {
        state.lineComment = true
        while i < u.count, !rawTextEnd(at: i) {
          if i % 4096 == 0 { try Task.checkCancellation() }
          i += 1
        }
        emit(start, .comment)
        continue
      }
      if has("/*", at: i), !hashComments, lang != "json" {
        state.blockEnd = "*/"
        i += 2
        emit(start, .comment)
        continue
      }
      if u[i] == 34 || u[i] == 39 || u[i] == 96 {
        if lang == "rust", u[i] == 39, i + 1 < u.count, identifierStart(u[i + 1]) {
          var end = i + 2
          while end < u.count, identifierStart(u[end]) || (48...57).contains(u[end]) { end += 1 }
          if end == u.count || u[end] != 39 {
            i = end
            emit(start, .keyword)
            continue
          }
        }
        state.quote = u[i]
        state.tripleQuote =
          ["python", "swift", "kotlin"].contains(lang) && i + 2 < u.count && u[i + 1] == u[i]
          && u[i + 2] == u[i]
        i += state.tripleQuote ? 3 : 1
        emit(start, .string)
        continue
      }
      if (48...57).contains(u[i]) {
        i += 1
        while i < u.count, word(u[i]) || u[i] == 46 { i += 1 }
        emit(start, .number)
        continue
      }
      if word(u[i]) || lang == "css" && u[i] == 45 {
        i += 1
        while i < u.count, word(u[i]) || lang == "css" && u[i] == 45 { i += 1 }
        let value = String(decoding: u[start..<i], as: UTF16.self)
        remember(value)
        var next = i
        while next < u.count, u[next] == 32 || u[next] == 9 { next += 1 }
        if lang == "css", next < u.count, u[next] == 58 {
          emit(start, .attribute)
        } else if keywords.contains(value)
          || ["sql", "cypher"].contains(lang) && keywords.contains(value.uppercased())
        {
          emit(start, .keyword)
        }
        continue
      }
      if [40, 41, 91, 93, 123, 125].contains(u[i]) {
        i += 1
        emit(start, .bracket)
      } else {
        i += 1
      }
    }
    // Ordinary code strings cannot carry to another line without an escape;
    // template strings and markup attributes can.
    if !state.tag, state.quote != 96, !state.tripleQuote, language != "bash",
      u.last == 10 || u.last == 13
    {
      let content =
        u.last == 10 ? Array(u.dropLast(u.count > 1 && u[u.count - 2] == 13 ? 2 : 1)) : u
      if content.last != 92 { state.quote = 0 }
    }
    return Lexed(tokens: tokens, words: words.sorted())
  }
}
