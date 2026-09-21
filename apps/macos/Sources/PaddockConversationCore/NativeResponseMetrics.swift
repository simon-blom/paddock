import Foundation

/// Native counterpart of useChatStream's phase accounting. Tool/discovery
/// events are not tokens, and reasoning time is never answer decode time.
struct NativeResponseMetrics: Sendable {
  typealias V = ConversationValue
  typealias O = [String: V]
  var firstToken: Double?
  var reasoningStart: Double?
  var answerStart: Double?
  var terminalAt: Double?

  mutating func observe(_ event: O, seconds: Double) {
    guard seconds.isFinite, seconds >= 0 else { return }
    if ["response.completed", "response.incomplete", "response.failed"].contains(
      event["type"]?.string ?? "")
    {
      if terminalAt == nil { terminalAt = seconds }
      return
    }
    guard event["delta"]?.string?.isEmpty == false else { return }
    switch event["type"]?.string {
    case "response.output_text.delta", "response.refusal.delta":
      if firstToken == nil { firstToken = seconds }
      if answerStart == nil { answerStart = seconds }
    case "response.reasoning_text.delta", "response.reasoning_summary_text.delta",
      "response.reasoning.delta":
      if firstToken == nil { firstToken = seconds }
      if reasoningStart == nil { reasoningStart = seconds }
    default: break
    }
  }

  func usage(_ response: O?, seconds: Double, previous: O? = nil) -> O? {
    guard let wire = response?["usage"]?.object else { return previous }
    // A checkpoint/save after the terminal frame is not model decode time.
    let seconds = terminalAt ?? seconds
    let previous = previous ?? [:]
    let reasoning = max(0, wire["output_tokens_details"]?["reasoning_tokens"]?.integer ?? 0)
    let answer = wire["output_tokens"]?.integer.map { max(0, $0 - reasoning) }
    var result: O = [:]
    func set(_ key: String, _ value: Double?) {
      if let value, value.isFinite, value >= 0 { result[key] = .number(Decimal(value)) }
    }
    result["promptTokens"] = wire["input_tokens"] ?? previous["promptTokens"]
    if let answer {
      result["completionTokens"] = .number(
        Decimal(answer + (previous["completionTokens"]?.integer ?? 0)))
    }
    let rt = reasoning + (previous["reasoningTokens"]?.integer ?? 0)
    if rt > 0 { result["reasoningTokens"] = .number(Decimal(rt)) }
    set("ms", seconds * 1000 + (previous["ms"]?.double ?? 0))
    set("ttftMs", firstToken.map { ($0 * 1000).rounded() } ?? previous["ttftMs"]?.double)
    // A stream can buffer entire tool arguments, and reasoning can resume
    // after tool execution. Delta arrival times cannot partition engine time.
    let timing = response?["paddock_timing"]?.object
    let hasPrevious =
      (previous["completionTokens"]?.integer ?? 0) + (previous["reasoningTokens"]?.integer ?? 0) > 0
    if timing?["source"]?.string == "engine", timing?["version"]?.integer == 1,
      !hasPrevious || previous["timingSource"]?.string == "engine",
      let decode = timing?["decode_ms"]?.double, decode.isFinite, decode >= 0
    {
      set("decodeMs", decode + (previous["decodeMs"]?.double ?? 0))
      for (key, wireKey) in [("prefillMs", "prefill_ms"), ("queueMs", "queue_ms")] {
        if let n = timing?[wireKey]?.double { set(key, n + (previous[key]?.double ?? 0)) }
      }
      result["timingSource"] = .string("engine")
    } else {
      result["timingSource"] = .string("end-to-end")
    }
    result = Self.presentation(result)
    if wire["cost"]?.double != nil || previous["costUsd"]?.double != nil {
      set("costUsd", (wire["cost"]?.double ?? 0) + (previous["costUsd"]?.double ?? 0))
    }
    result["provider"] = response?["provider"]?.string.map(V.string) ?? previous["provider"]
    return result
  }

  /// Also repairs legacy display math without rewriting conversations. Only
  /// validated engine timing may claim decode speed; speech keeps its own rate.
  static func presentation(_ usage: O) -> O {
    guard !usage.isEmpty else { return usage }
    var u = usage
    for key in ["tps", "answerMs", "reasoningMs", "reasoningTps"] { u.removeValue(forKey: key) }
    let total = (u["completionTokens"]?.double ?? 0) + (u["reasoningTokens"]?.double ?? 0)
    let engine = u["timingSource"]?.string == "engine"
    let duration = engine ? u["decodeMs"]?.double : u["ms"]?.double
    u["timingSource"] = .string(engine ? "engine" : "end-to-end")
    if let duration, duration.isFinite, duration > 0, total.isFinite, total > 0 {
      u["tps"] = .number(Decimal(total * 1000 / duration))
    }
    return u
  }
}
