import Foundation
import PaddockClient

enum CloudOrder: String, CaseIterable, Identifiable {
  case newest, oldest, trending, name, nameDescending, cheapest, expensive, largestContext,
    smallestContext
  var id: Self { self }
  var title: String {
    switch self {
    case .newest: "Published · newest first"
    case .oldest: "Published · oldest first"
    case .trending: "Trending"
    case .name: "Name · A-Z"
    case .nameDescending: "Name · Z-A"
    case .cheapest: "Token price · lowest first"
    case .expensive: "Token price · highest first"
    case .largestContext: "Context · largest first"
    case .smallestContext: "Context · smallest first"
    }
  }
}

enum CloudFeature: String, CaseIterable, Identifiable {
  case vision = "Vision"
  case reasoning = "Thinking"
  case tools = "Tools"
  case speech = "Speech"
  case free = "Free"
  var id: Self { self }
  var symbol: String {
    switch self {
    case .vision: "photo"
    case .reasoning: "brain"
    case .tools: "wrench.and.screwdriver"
    case .speech: "waveform"
    case .free: "tag"
    }
  }
  func matches(_ model: CloudModel) -> Bool {
    switch self {
    case .vision: model.vision == true
    case .reasoning: model.reasoning == true
    case .tools: model.tools == true
    case .speech: model.asr == true
    case .free: model.free == true
    }
  }
}

/// UI-only ordering and formatting over the shared Rust projection. No billing
/// calculation or second persistent registry belongs here.
enum CloudCatalogPresentation {
  static func orders(for models: [CloudModel], ranked: Bool) -> [CloudOrder] {
    CloudOrder.allCases.filter { order in
      switch order {
      case .newest, .oldest: models.contains { publication($0) != nil }
      case .trending: ranked
      case .cheapest, .expensive:
        models.contains {
          !audioRate($0)
            && (validPrice($0.promptPrice) != nil || validPrice($0.completionPrice) != nil)
        }
      case .largestContext, .smallestContext: models.contains { ($0.ctx ?? 0) > 0 }
      case .name, .nameDescending: true
      }
    }
  }
  static func manualPick(_ query: String) -> CloudModelPick? {
    let id = query.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !id.isEmpty, id.utf8.count <= 256, !id.contains("@"), !id.hasPrefix("cloud:"),
      !id.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains)
    else { return nil }
    return CloudModelPick(id: id)
  }
  private static let aliases: [String: [String]] = [
    "claude": ["anthropic"], "gpt": ["openai"], "oai": ["openai"], "gemini": ["google"],
    "gemma": ["google"], "llama": ["meta"], "grok": ["x-ai"], "kimi": ["moonshotai"],
    "glm": ["z-ai"], "qwen": ["alibaba"],
  ]
  private static let speechWords: Set<String> = [
    "speech", "voice", "audio", "transcribe", "transcription", "stt", "asr",
  ]

  static func vendor(_ model: CloudModel) -> String? {
    vendor(id: model.id)
  }
  static func vendor(id modelID: String) -> String? {
    CloudModelIdentity.vendor(modelID)
  }
  static func name(_ model: CloudModel) -> String {
    CloudModelIdentity.resolve(id: model.id, display: model.display).name
  }
  static func audioRate(_ model: CloudModel) -> Bool {
    model.asr == true && (model.completionPrice ?? 0) == 0
  }
  static func validPrice(_ value: Double?) -> Double? {
    guard let value, value.isFinite, value >= 0 else { return nil }
    return value
  }
  static func dollars(_ value: Double?) -> String {
    guard let value = validPrice(value) else { return "Not listed" }
    if value == 0 { return "$0" }
    if value < 0.000001 { return String(format: "$%.3g", value) }
    return "$"
      + value.formatted(
        .number.locale(Locale(identifier: "en_US")).precision(.fractionLength(0...6)))
  }
  static func perMillion(_ value: Double?) -> String {
    dollars(validPrice(value).map { $0 * 1_000_000 })
  }
  static func priceSummary(_ model: CloudModel) -> String {
    if audioRate(model) { return "\(dollars(model.promptPrice)) · audio rate" }
    if validPrice(model.promptPrice) == nil && validPrice(model.completionPrice) == nil {
      return "Pricing not listed"
    }
    return "\(perMillion(model.promptPrice)) in · \(perMillion(model.completionPrice)) out / M"
  }
  static func tokens(_ value: UInt64?) -> String {
    guard let value, value > 0 else { return "Not listed" }
    if value >= 1_000_000 {
      return "\((Double(value) / 1_000_000).formatted(.number.precision(.fractionLength(0...1))))M"
    }
    if value >= 1000 {
      return "\((Double(value) / 1000).formatted(.number.precision(.fractionLength(0...1))))K"
    }
    return String(value)
  }
  static func publication(_ model: CloudModel) -> String? {
    guard let value = model.created, value > 0, value <= 253_402_300_799 else { return nil }
    return Date(timeIntervalSince1970: Double(value)).formatted(
      .iso8601.year().month().day().dateSeparator(.dash))
  }
  static func modelURL(_ id: String) -> URL? {
    let parts = id.split(separator: "/", omittingEmptySubsequences: false)
    guard id.utf8.count <= 256, parts.count == 2,
      parts.allSatisfy({ part in
        !part.isEmpty && part != "." && part != ".."
          && part.utf8.allSatisfy {
            (65...90).contains($0) || (97...122).contains($0)
              || (48...57).contains($0) || [45, 95, 46, 58, 126].contains($0)
          }
      })
    else { return nil }
    return URL(string: "https://openrouter.ai/\(id)")
  }

  /// Multi-token AND, the web picker's maker aliases, word-boundary ranking,
  /// and searchable speech capability. Search relevance precedes chosen order.
  static func entries(
    _ models: [CloudModel], query: String, filters: Set<CloudFeature>, order: CloudOrder
  ) -> [CloudModel] {
    let tokens = query.lowercased().split(whereSeparator: \.isWhitespace).map(String.init)
    func number(_ model: CloudModel) -> Double? {
      switch order {
      case .newest, .oldest:
        return model.created.flatMap { $0 > 0 && $0 <= 253_402_300_799 ? Double($0) : nil }
      case .cheapest, .expensive:
        // Unknown input/output prices are not zero; audio duration rates are
        // incomparable. Both unknown and partial token prices stay at the end.
        guard !audioRate(model), let input = validPrice(model.promptPrice),
          let output = validPrice(model.completionPrice)
        else { return nil }
        return validPrice(input + output)
      case .largestContext, .smallestContext: return model.ctx.flatMap { $0 > 0 ? Double($0) : nil }
      default: return nil
      }
    }
    // Build sort keys once per row, never format dates or run name regexes
    // inside the comparator on each keystroke.
    let rows = models.enumerated().compactMap {
      index, model -> (model: CloudModel, index: Int, score: Double, name: String, number: Double?)?
      in
      guard filters.allSatisfy({ $0.matches(model) }) else { return nil }
      let scores = tokens.map { score(model, token: $0) }
      guard scores.allSatisfy({ $0 > 0 }) else { return nil }
      return (model, index, scores.reduce(0, +), name(model), number(model))
    }
    return rows.sorted { lhs, rhs in
      if lhs.score != rhs.score { return lhs.score > rhs.score }
      switch order {
      case .trending: break
      case .name, .nameDescending:
        let result = lhs.name.localizedStandardCompare(rhs.name)
        if result != .orderedSame {
          return result == (order == .name ? .orderedAscending : .orderedDescending)
        }
      default:
        let a = lhs.number
        let b = rhs.number
        if let a, let b, a != b {
          return [.newest, .expensive, .largestContext].contains(order) ? a > b : a < b
        }
        if (a == nil) != (b == nil) { return a != nil }
      }
      return lhs.index < rhs.index
    }.map(\.model)
  }

  private static func score(_ model: CloudModel, token: String) -> Double {
    let names = [model.id.lowercased(), (model.display ?? "").lowercased()]
    var best: Double = model.asr == true && speechWords.contains(token) ? 2 : 0
    for term in [token] + (aliases[token] ?? []) {
      if names.contains(where: { $0.hasPrefix(term) }) { return 3 }
      if names.contains(where: { name in
        name.components(separatedBy: CharacterSet(charactersIn: "/-_ .:")).contains {
          $0.hasPrefix(term)
        }
      }) {
        best = max(best, 2)
      } else if names.contains(where: { $0.contains(term) }) {
        best = max(best, 1)
      } else if model.blurb?.lowercased().contains(term) == true {
        best = max(best, 0.5)
      }
    }
    return best
  }
}
