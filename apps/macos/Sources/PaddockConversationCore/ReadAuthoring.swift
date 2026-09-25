import Foundation

extension ReadQuestion {
  public static func uniqueID(_ base: String, taken: [String]) -> String {
    let used = Set(taken)
    if !used.contains(base) { return base }
    var n = 2
    while used.contains("\(base)_\(n)") { n += 1 }
    return "\(base)_\(n)"
  }

  // Keep this rule identical to studio/src/lib/reads.ts: IDs are prompt text,
  // not just opaque UI identities.
  public static func derivedID(_ text: String, taken: [String] = []) -> String {
    let stop = Set(
      ("a an the is are was were be been being do does did of to in on at for with and or "
        + "this that these those it its what which how who whom there here any from by as into "
        + "about than then text state message following above below please rate decide whether "
        + "if has have had will would should could can not no yes you your we our they their he "
        + "she him her his them i me my one").split(separator: " ").map(String.init))
    let words = text.lowercased().replacingOccurrences(
      of: #"[^a-z0-9\s_-]+"#, with: " ", options: .regularExpression
    ).split(whereSeparator: \.isWhitespace).map(String.init)
    let content = Array(words.filter { !stop.contains($0) }.prefix(3))
    let trim = CharacterSet(charactersIn: "_-")
    let base = String(
      (content.isEmpty ? Array(words.suffix(2)) : content)
        .joined(separator: "_").trimmingCharacters(in: trim).prefix(32)
    )
    .trimmingCharacters(in: trim)
    return uniqueID(base.isEmpty ? "q" : base, taken: taken)
  }

  public static func cleanID(_ text: String) -> String {
    text.replacingOccurrences(of: #"[\s:]+"#, with: "_", options: .regularExpression)
  }
}

extension ReadDraft {
  public static var example: ReadDraft {
    var draft = ReadDraft()
    draft.state = """
      Subject: Portal down again

      Hi, this is the third time this week the customer portal has gone down during business hours. We have 40 agents unable to log in right now and customers are calling. I need someone on this immediately - we are paying for the enterprise tier and this is unacceptable. Please call me back on the number on file.

      - Dana, Ops lead at Northwind
      """
    var urgent = ReadQuestion(questionID: "")
    urgent.instructions = "Does this message need action within the hour?"
    urgent.yesMeans = "an outage or blocker affecting many people now"
    urgent.noMeans = "a request that can wait a day"
    var topic = ReadQuestion(questionID: "", kind: .choice)
    topic.instructions = "What is this message about?"
    topic.options = [
      .init(name: "outage", description: "a service is down or broken"),
      .init(name: "billing", description: "invoices, plans or payment"),
      .init(name: "feature", description: "a request for something new"),
      .init(name: "other", description: "none of these"),
    ]
    var mood = ReadQuestion(questionID: "", kind: .score)
    mood.instructions = "How upset is the sender?"
    mood.levels = ["calm", "annoyed", "furious"].map { .init(name: $0) }
    draft.questions = [urgent, topic, mood]
    for index in draft.questions.indices {
      draft.questions[index].questionID = ReadQuestion.derivedID(
        draft.questions[index].instructions,
        taken: draft.questions.prefix(index).map(\.questionID))
    }
    return draft
  }
  /// JSON objects are unordered in ConversationValue. Keep authored question
  /// and option order in saved sets and exports without changing chat's codec.
  public func orderedJSON(model: String? = nil) throws -> String {
    func quoted(_ text: String) throws -> String { try Self.json(.string(text)) }
    let rows = try questions.map { q in
      var fields = q.wire.object ?? [:]
      fields.removeValue(forKey: "criteria")
      var members = try fields.keys.sorted().map { key in
        try quoted(key) + ": " + Self.json(fields[key]!)
      }
      if q.kind == .choice {
        let options = try q.options.map {
          try quoted($0.name.trimmingCharacters(in: .whitespacesAndNewlines)) + ": "
            + quoted($0.description.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        members.append("\"criteria\": {" + options.joined(separator: ", ") + "}")
      } else if let criteria = q.wire["criteria"] {
        members.append("\"criteria\": " + (try Self.json(criteria)))
      }
      return try "    " + quoted(q.questionID) + ": {" + members.joined(separator: ", ") + "}"
    }
    var fields = [
      "  \"samples\": " + (try Self.json(setBody["samples"]!)),
      "  \"questions\": {\n" + rows.joined(separator: ",\n") + "\n  }",
    ]
    if let model {
      fields.insert(try "  \"model\": " + quoted(model), at: 0)
      fields.insert(try "  \"state\": " + quoted(state), at: 1)
    }
    return "{\n" + fields.joined(separator: ",\n") + "\n}"
  }
}

/// An order index over already JSONDecoder-validated UTF-8. It doesn't decode
/// values a second time, reject valid escaped keys, or guess order with regex.
/// Bound recursion and reject duplicate keys instead of silently losing rows.
struct ReadJSONOrder {
  var keys: [[String]: [String]] = [:]
  private var bytes: [UInt8]
  private var index = 0
  init(_ data: Data) throws {
    bytes = Array(data)
    try visit([], depth: 0)
  }
  private mutating func whitespace() {
    while index < bytes.count && [9, 10, 13, 32].contains(bytes[index]) { index += 1 }
  }
  private mutating func string() throws -> String {
    let start = index
    index += 1
    while index < bytes.count {
      let byte = bytes[index]
      index += 1
      if byte == 92 {
        index += 1
      } else if byte == 34 {
        return try JSONDecoder().decode(String.self, from: Data(bytes[start..<index]))
      }
    }
    throw ConversationFailure.invalid("Unfinished JSON string.")
  }
  private mutating func visit(_ path: [String], depth: Int) throws {
    guard depth < 64 else { throw ConversationFailure.invalid("JSON is nested too deeply.") }
    whitespace()
    guard index < bytes.count else { throw ConversationFailure.invalid("Incomplete JSON.") }
    switch bytes[index] {
    case 123:
      index += 1
      whitespace()
      var names: [String] = []
      while index < bytes.count && bytes[index] != 125 {
        let key = try string()
        guard !names.contains(key) else {
          throw ConversationFailure.invalid("Duplicate JSON key: \(key)")
        }
        names.append(key)
        whitespace()
        index += 1  // colon; syntax was checked by JSONDecoder
        try visit(path + [key], depth: depth + 1)
        whitespace()
        if index < bytes.count && bytes[index] == 44 {
          index += 1
          whitespace()
        }
      }
      keys[path] = names
      index += 1
    case 91:
      index += 1
      whitespace()
      var element = 0
      while index < bytes.count && bytes[index] != 93 {
        try visit(path + [String(element)], depth: depth + 1)
        element += 1
        whitespace()
        if index < bytes.count && bytes[index] == 44 {
          index += 1
          whitespace()
        }
      }
      index += 1
    case 34: _ = try string()
    default:
      while index < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[index]) {
        index += 1
      }
    }
  }
}
