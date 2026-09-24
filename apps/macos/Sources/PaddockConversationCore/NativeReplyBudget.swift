import Foundation

/// Native counterpart of Studio's tokens.ts windowRemaining policy. A missing
/// user cap means available context, sent as an explicit max_output_tokens so
/// the run record shows what rode (the runner's own default for an omitted cap
/// has been the window too since 2026-09-22; it was a flat 1024 before).
enum NativeReplyBudget {
  typealias V = ConversationValue
  /// One lane's effective ceiling. Unknown capacity is nil (leave the field
  /// out and let the endpoint choose), not a guessed 4096-token maximum.
  /// A custom preference is an upper bound, so it too respects model limits.
  static func resolve(
    requested: Int?, cloud: Bool, context: Int, prompt: Int = 0,
    outputCeiling: Int? = nil, exact: Int = 0
  ) -> Int? {
    let ceiling = outputCeiling.flatMap { $0 > 0 ? $0 : nil }
    var available =
      context > 0
      ? (cloud
        ? maximum(context: context, prompt: prompt, outputCeiling: ceiling, exact: exact)
        : context) : ceiling
    if let limit = available, let ceiling { available = min(limit, ceiling) }
    guard let requested else { return available }
    return available.map { min(requested, $0) } ?? requested
  }
  /// Paddock clamps against the exactly tokenized prompt in service.rs (also
  /// after multimodal admission). Do not waste another 1024 tokens on a client
  /// estimate.
  static func localMaximum(context: Int) -> Int { context > 0 ? context : 4096 }
  /// Below this much real room a turn is refused rather than sent.
  static let minimumUsefulReply = 256
  /// `exact` is the part of the prompt a server has already counted and
  /// `prompt` the part that is still an estimate; the margin is charged to the
  /// estimate only. The floor borrows from that margin and never from the
  /// window, and 0 means refuse the turn. Same rules, same fixture, as
  /// tokens.ts windowRemaining.
  static func maximum(context: Int, prompt: Int, outputCeiling: Int? = nil, exact: Int = 0) -> Int {
    let ceiling = outputCeiling.flatMap { $0 > 0 ? $0 : nil } ?? Int.max
    guard context > 0 else { return min(4096, ceiling) }
    let room = context - exact - prompt
    guard room >= minimumUsefulReply else { return 0 }
    let slack = max(1024, Int((Double(context) * 0.02).rounded()))
    let margin = exact > 0 ? max(128, (prompt + 3) / 4) : slack
    return min(ceiling, max(room - margin, min(512, room)))
  }

  /// Estimate the actual lane sent, not discarded branches or other Compare
  /// lanes. Never count base64 attachment bytes as text tokens. Like web Studio,
  /// this is a guardrail; the runner owns exact tokenization and context limits.
  static func estimate(input: [V], instructions: String) -> Int {
    func text(_ s: String) -> Int { (s.utf16.count + 3) / 4 }
    func parts(_ value: V?) -> Int {
      if let s = value?.string { return text(s) }
      return (value?.array ?? []).reduce(0) { total, part in
        switch part["type"]?.string {
        case "input_text", "output_text", "reasoning_text":
          return total + text(part["text"]?.string ?? "")
        case "input_image": return total + 512
        default: return total
        }
      }
    }
    return (instructions.isEmpty ? 0 : text(instructions) + 4)
      + input.reduce(0) { total, item in
        total + 4 + parts(item["content"])
      }
  }
}
