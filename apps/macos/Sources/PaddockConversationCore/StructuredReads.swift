import Foundation

/// The same /v1/systemone wire contract as studio/src/lib/reads.ts. No chat
/// generation, inferred confidence, hidden prompt, or cloud-model fallback.
public struct ReadQuestion: Identifiable, Equatable, Codable, Sendable {
  public enum Kind: String, CaseIterable, Codable, Sendable {
    case noul, choice, score
    public var title: String {
      switch self {
      case .noul: "Yes / no"
      case .choice: "Choice"
      case .score: "Score"
      }
    }
  }
  public struct Option: Identifiable, Equatable, Codable, Sendable {
    public var id = UUID()
    public var name = ""
    public var description = ""
    public init(name: String = "", description: String = "") {
      self.name = name
      self.description = description
    }
  }
  public var id = UUID()
  public var questionID: String
  public var idTouched = false
  public var kind: Kind
  public var instructions = ""
  public var yesMeans = ""
  public var noMeans = ""
  // Retain the other type's fields when switching types, like the web editor.
  public var options = [Option(), Option()]
  public var levels = [Option(), Option(), Option()]
  public init(questionID: String, kind: Kind = .noul) {
    self.questionID = questionID
    self.kind = kind
  }
  public var validation: String? {
    if questionID.isEmpty || questionID.contains(where: { $0.isWhitespace || $0 == ":" }) {
      return "Use a question ID without spaces or a colon."
    }
    let entries = kind == .choice ? options : kind == .score ? levels : []
    if kind != .noul {
      let names = entries.map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines) }
      if !(2...26).contains(names.count) { return "Use between 2 and 26 options or levels." }
      if names.contains("") { return "Every option or level needs a name." }
      if Set(names).count != names.count { return "Option and level names must differ." }
    }
    return nil
  }
  public var wire: ConversationValue {
    var q: [String: ConversationValue] = [
      "type": .string(kind.rawValue),
      "instructions": .string(instructions.trimmingCharacters(in: .whitespacesAndNewlines)),
    ]
    switch kind {
    case .noul:
      var criteria: [String: ConversationValue] = [:]
      let yes = yesMeans.trimmingCharacters(in: .whitespacesAndNewlines)
      let no = noMeans.trimmingCharacters(in: .whitespacesAndNewlines)
      if !yes.isEmpty { criteria["true"] = .string(yes) }
      if !no.isEmpty { criteria["false"] = .string(no) }
      if !criteria.isEmpty { q["criteria"] = .object(criteria) }
    case .choice:
      // Invalid duplicate names are diagnosed before sending; never trap here
      // while a user is editing two temporarily identical fields.
      q["criteria"] = .object(
        Dictionary(
          options.map {
            (
              $0.name.trimmingCharacters(in: .whitespacesAndNewlines),
              ConversationValue.string(
                $0.description.trimmingCharacters(in: .whitespacesAndNewlines))
            )
          }, uniquingKeysWith: { _, last in last }))
    case .score:
      q["criteria"] = .array(
        levels.map { .string($0.name.trimmingCharacters(in: .whitespacesAndNewlines)) })
    }
    return .object(q)
  }
}

public struct ReadDraft: Equatable, Sendable {
  public var state = ""
  public var questions = [ReadQuestion(questionID: "q1")]
  /// 0 is auto locally; wire format always sends "auto", never 0.
  public var samples = 0
  public init() {}
  public func validation(maxQuestions: Int = 64, maxSamples: Int = 32) -> String? {
    if questions.isEmpty || questions.count > maxQuestions {
      return "Use 1 to \(maxQuestions) questions."
    }
    if !(0...maxSamples).contains(samples) {
      return "Use automatic reads or 1 to \(maxSamples) reads."
    }
    if Set(questions.map(\.questionID)).count != questions.count {
      return "Question IDs must differ."
    }
    return questions.compactMap(\.validation).first
  }
  public var setBody: ConversationValue {
    .object([
      "samples": samples == 0 ? .string("auto") : .number(Decimal(samples)),
      "questions": .object(
        Dictionary(
          questions.map { ($0.questionID, $0.wire) }, uniquingKeysWith: { _, last in last })),
    ])
  }
  public var ordering: [[String]] {
    questions.map { [$0.questionID] + ($0.kind == .choice ? $0.options.map(\.name) : []) }
  }
  public func request(model: String) -> ConversationValue {
    var v = setBody.object!
    v["model"] = .string(model)
    v["state"] = .string(state)
    return .object(v)
  }
  public static func parse(_ data: Data) throws -> ReadDraft {
    guard data.count <= 512 * 1024 else { throw ConversationFailure.tooLarge }
    let value = try JSONDecoder().decode(ConversationValue.self, from: data)
    guard let root = value.object, let map = (root["questions"] ?? value).object else {
      throw ConversationFailure.invalid("Use a questions object or a complete read request.")
    }
    var draft = ReadDraft()
    if let state = root["state"]?.string { draft.state = state }
    if let s = root["samples"] {
      if s == .string("auto") {
        draft.samples = 0
      } else if let n = s.integer, n > 0 {
        draft.samples = n
      } else {
        throw ConversationFailure.invalid("Reads must be auto or a positive integer.")
      }
    }
    let order = try ReadJSONOrder(data)
    let path = root["questions"] == nil ? [] : ["questions"]
    draft.questions = try (order.keys[path] ?? []).map { id in
      let raw = map[id]!
      let type = raw["type"]?.string ?? ""
      guard
        let kind = ReadQuestion.Kind(rawValue: ["bool", "boolean"].contains(type) ? "noul" : type)
      else {
        throw ConversationFailure.invalid("\(id): unsupported question type.")
      }
      var q = ReadQuestion(questionID: id, kind: kind)
      q.idTouched = true
      q.instructions = raw["instructions"]?.string ?? ""
      let criteria = raw["criteria"]
      switch kind {
      case .noul:
        q.yesMeans = criteria?["true"]?.string ?? ""
        q.noMeans = criteria?["false"]?.string ?? ""
      case .choice:
        guard let options = criteria?.object else {
          throw ConversationFailure.invalid("\(id): choice criteria must be an object.")
        }
        q.options = try (order.keys[path + [id, "criteria"]] ?? []).map { name in
          guard let text = options[name]?.string else {
            throw ConversationFailure.invalid("\(id): option descriptions must be text.")
          }
          return .init(name: name, description: text)
        }
      case .score:
        guard let levels = criteria?.array else {
          throw ConversationFailure.invalid("\(id): score criteria must be an ordered array.")
        }
        q.levels = try levels.map {
          guard let name = $0.string else {
            throw ConversationFailure.invalid("\(id): level names must be text.")
          }
          return .init(name: name)
        }
      }
      return q
    }
    if let error = draft.validation() { throw ConversationFailure.invalid(error) }
    return draft
  }
  public static func json(_ value: ConversationValue) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(value), as: UTF8.self)
  }
}

public struct ReadResponse: Decodable, Sendable {
  public struct Answer: Decodable, Sendable {
    public let type: String
    public let confidence: Double
    public let agreement: Double
    public let outside: Double
    public let noul: Double?
    public let choice: String?
    public let score: Double?
    public let level: String?
    public let probabilities: [String: Double]?
    public let legend: [String: String]?
    public var label: String {
      if let noul { return noul >= 0.5 ? "Yes" : "No" }
      return choice ?? level ?? "—"
    }
    public var bars: [(name: String, probability: Double)] {
      if let noul { return [("Yes", noul), ("No", 1 - noul)] }
      if type == "score", let legend {
        return legend.keys.sorted { (Int($0) ?? 0) < (Int($1) ?? 0) }
          .map { (legend[$0]!, probabilities?[$0] ?? 0) }
      }
      return (probabilities ?? [:]).sorted {
        $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value
      }
      .map { ($0.key, $0.value) }
    }
    public var nearTie: Bool {
      let probabilities = bars.map(\.probability).sorted(by: >)
      return probabilities.count >= 2 && probabilities[0] - probabilities[1] < 0.05
    }
    public var scorePosition: Double {
      guard let score, let legend, legend.count > 1 else { return 0 }
      return min(1, max(0, score / Double(legend.count - 1)))
    }
    public static func confidenceBin(_ value: Double) -> Int {
      if !(value >= 0.5) { return 0 }
      if value < 0.7 { return 1 }
      return value < 0.9 ? 2 : 3
    }
  }
  public struct Diagnostics: Decodable, Sendable {
    public struct Question: Decodable, Sendable {
      public struct Sample: Decodable, Sendable {
        public let pick: String
        public let confidence: Double
        public let entropy: Double
      }
      public let id: String
      public let label: String
      public let entropy: Double
      public let labelMass: Double
      public let position: Int?
      public let reads: [Sample]?
      private enum CodingKeys: String, CodingKey {
        case id, label, entropy, reads, position
        case labelMass = "label_mass"
      }
    }
    public struct Timing: Decodable, Sendable {
      public let totalMilliseconds: Double
      private enum CodingKeys: String, CodingKey { case totalMilliseconds = "total_ms" }
    }
    public let reads: Int
    public let canvas: Int
    public let questions: [Question]
    public let timing: Timing
  }
  public let model: String
  public let answers: [String: Answer]
  public let diagnostics: Diagnostics
  public struct Usage: Decodable, Sendable {
    public let inputTokens: Int?
    public let outputTokens: Int?
    private enum CodingKeys: String, CodingKey {
      case inputTokens = "input_tokens"
      case outputTokens = "output_tokens"
    }
  }
  public let usage: Usage?
  public func validate(for questions: [ReadQuestion]) throws {
    func probability(_ p: Double) -> Bool { p.isFinite && (0...1).contains(p) }
    guard Set(answers.keys) == Set(questions.map(\.questionID)),
      (1...32).contains(diagnostics.reads), diagnostics.canvas > 0,
      diagnostics.timing.totalMilliseconds.isFinite, diagnostics.timing.totalMilliseconds >= 0
    else {
      throw ConversationFailure.invalid("The runner returned an incomplete or invalid read result.")
    }
    for question in questions {
      let answer = answers[question.questionID]!
      guard answer.type == question.kind.rawValue,
        probability(answer.confidence), probability(answer.agreement), probability(answer.outside),
        answer.probabilities?.values.allSatisfy(probability) ?? true
      else {
        throw ConversationFailure.invalid("Invalid probabilities for \(question.questionID).")
      }
      let valid: Bool
      switch question.kind {
      case .noul: valid = answer.noul.map(probability) == true
      case .choice:
        valid =
          answer.choice.map { answer.probabilities?[$0] != nil } == true
          && Set(answer.probabilities?.keys.map { $0 } ?? [])
            == Set(question.options.map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines) })
      case .score:
        valid =
          answer.score.map { $0.isFinite && $0 >= 0 && $0 <= Double(question.levels.count - 1) }
          == true
          && answer.legend?.count == question.levels.count
          && answer.level.map { answer.legend?.values.contains($0) == true } == true
      }
      guard valid else {
        throw ConversationFailure.invalid("Invalid answer for \(question.questionID).")
      }
    }
  }
}
