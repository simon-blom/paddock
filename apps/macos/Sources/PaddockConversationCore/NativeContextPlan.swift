import CryptoKit
import Foundation

/// The native counterpart of tokens.ts. Raw history is never removed: this
/// plan only selects what rides in the next request. Shared fixtures pin the
/// window/summary arithmetic; request-prefix fingerprints gate exact reuse.
struct NativeContextPlan: Sendable, Equatable {
  typealias V = ConversationValue
  typealias O = [String: V]
  var from = 0
  var summary: String?
  var item: O?
  var threshold = 0

  static func tokens(_ text: String) -> Int { (text.utf16.count + 3) / 4 }
  static func messageTokens(_ message: O) -> Int {
    tokens(ConversationDocument.text(message)) + 4
      + (message["content"]?.array ?? []).filter { $0["type"]?.string == "image" }.count * 512
  }
  static func reserve(_ cap: V, context: Int = 0, outputCeiling: Int? = nil) -> Int {
    let headroom = context > 0 ? max(1, context / 4) : 4096
    return min(
      cap.integer.flatMap { $0 > 0 ? $0 : nil } ?? 4096, 4096, headroom,
      outputCeiling.flatMap { $0 > 0 ? $0 : nil } ?? Int.max)
  }
  static func summaryBlock(_ text: String) -> String {
    "Summary of the earlier part of this conversation (older messages were compacted):\n\(text)"
  }
  static func summaryValid(_ doc: ConversationDocument) -> Bool {
    let f = doc.fields
    guard let text = f["summary"]?.string, !text.isEmpty, text.utf8.count <= 64 * 1024,
      let count = f["summaryCount"]?.integer, count > 0, count <= doc.activeMessages.count,
      doc.activeMessages[count - 1]["id"] == f["summaryLastId"]
    else { return false }
    return f["summaryModel"] == nil || f["summaryModel"] == f["model"]
  }
  static func contextTokens(_ doc: ConversationDocument, draft: String = "") -> Int {
    let path = doc.activeMessages
    let extra = draft.isEmpty ? 0 : tokens(draft) + 4
    for (index, message) in path.enumerated().reversed() {
      guard message["role"]?.string == "assistant",
        let usage = message["usage"]?.object,
        let prompt = usage["promptTokens"]?.integer, (1...1_000_000_000).contains(prompt)
      else { continue }
      // Tool rounds and page runs report accumulated billing, not a context
      // size. Another Compare lane's tokenizer cannot anchor this lane either.
      guard message["group"] == nil, message["docRun"] == nil, message["transcript"] == nil,
        (message["toolCalls"]?.array ?? []).isEmpty,
        (message["webSearches"]?.array ?? []).isEmpty,
        message["model"] == nil || message["model"] == doc.fields["model"],
        message["run"]?["systemPrompt"]?.string ?? "" == doc.fields["systemPrompt"]?.string ?? ""
      else { break }
      let answer = min(1_000_000_000, max(0, usage["completionTokens"]?.integer ?? 0))
      return prompt + answer + 4 + path.dropFirst(index + 1).reduce(0) { $0 + messageTokens($1) }
        + extra
    }
    let system = doc.fields["systemPrompt"]?.string ?? ""
    return (system.isEmpty ? 0 : tokens(system) + 4)
      + path.reduce(0) { $0 + messageTokens($1) } + extra
  }
  static func trim(_ doc: ConversationDocument, context: Int, reply: Int, extra: Int = 0) -> Int {
    let budget = context - reply - 1024 - extra
    guard context > 0, budget > 0 else { return 0 }
    let path = doc.activeMessages
    let system = doc.fields["systemPrompt"]?.string ?? ""
    var used = system.isEmpty ? 0 : tokens(system) + 4
    var first = max(0, path.count - 1)
    for (index, message) in path.enumerated().reversed() {
      let cost = messageTokens(message)
      if index < path.count - 1 && used + cost > budget { break }
      used += cost
      first = index
    }
    return first
  }
  static func serverThreshold(context: Int, reply: Int) -> Int {
    let budget = context - reply - 1024
    return budget > 0 ? max(512, Int(Double(budget) * 0.7)) : 0
  }
  static func resolve(
    _ doc: ConversationDocument, context: Int, reply: Int, summarize: Bool, server: Bool
  ) -> Self {
    var plan = Self()
    let path = doc.activeMessages
    if server && summarize {
      plan.threshold = serverThreshold(context: context, reply: reply)
      if plan.threshold > 0, let sc = doc.fields["serverCompaction"]?.object,
        let id = sc["id"]?.string, !id.isEmpty,
        let content = sc["content"]?.string, !content.isEmpty, content.utf8.count <= 1024 * 1024,
        sc["model"] == nil || sc["model"] == doc.fields["model"],
        let anchor = path.firstIndex(where: { $0["id"] == sc["tailStartId"] })
      {
        plan.from = anchor
        plan.item = [
          "type": .string("compaction"), "id": .string(id), "encrypted_content": .string(content),
        ]
        return plan
      }
    }
    plan.from = trim(doc, context: context, reply: reply)
    if plan.threshold == 0 && summarize && summaryValid(doc) && plan.from > 0,
      let summary = doc.fields["summary"]?.string
    {
      plan.summary = summary
      plan.from = max(
        doc.fields["summaryCount"]!.integer!,
        trim(doc, context: context, reply: reply, extra: tokens(summary) + 4))
    }
    // The pending assistant is not the user's request. Never trim a large
    // newest user turn merely because an empty response placeholder follows it.
    if let user = path.lastIndex(where: { $0["role"]?.string == "user" }) {
      plan.from = min(plan.from, user)
    }
    return plan
  }
  static func compactionTarget(_ doc: ConversationDocument, context: Int, reply: Int) -> Int {
    let path = doc.activeMessages
    let budget = context - reply - 1024
    guard context > 0, path.count >= 4, budget > 0,
      Double(contextTokens(doc)) >= Double(budget) * 0.7
    else { return 0 }
    var used = 0
    var keep = path.count
    for (index, message) in path.enumerated().reversed() {
      used += messageTokens(message)
      if Double(used) > Double(budget) * 0.35 && keep < path.count { break }
      keep = index
    }
    let target = min(keep, path.count - 2)
    let prior = summaryValid(doc) ? doc.fields["summaryCount"]!.integer! : 0
    return target > prior ? target : 0
  }

  /// Exact counts only reuse a byte-identical text prefix and identical
  /// instructions/tools/reasoning settings. Large or multimodal inputs fall
  /// back to estimation; never hash base64 media or retain a second prompt.
  static func prefixKey(_ body: O, count: Int) -> String? {
    guard let input = body["input"]?.array, (0...input.count).contains(count) else { return nil }
    let prefix = Array(input.prefix(count))
    var bytes = 0
    for item in prefix {
      guard item["type"]?.string == "message" || item["type"]?.string == "reasoning" else {
        return nil
      }
      if let text = item["content"]?.string { bytes += text.utf8.count }
      for part in item["content"]?.array ?? [] {
        guard ["input_text", "output_text", "reasoning_text"].contains(part["type"]?.string ?? "")
        else { return nil }
        bytes += part["text"]?.string?.utf8.count ?? 0
      }
      guard bytes <= 512 * 1024 else { return nil }
    }
    var value: O = ["input": .array(prefix)]
    for key in [
      "model", "instructions", "tools", "reasoning", "chat_template_kwargs", "file_metadata",
      "forensics",
    ] {
      value[key] = body[key]
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    guard let data = try? encoder.encode(value), data.count <= 1024 * 1024 else { return nil }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  static func replyPrompt(_ doc: ConversationDocument, body: O, model: String, pending: String) -> (
    exact: Int, estimated: Int
  ) {
    let input = body["input"]?.array ?? []
    let fallback = NativeReplyBudget.estimate(
      input: input, instructions: body["instructions"]?.string ?? "")
    for message in doc.activeMessages.reversed() where message["role"]?.string == "assistant" {
      if message["id"]?.string == pending && message["usage"] == nil { continue }
      guard message["group"] == nil, message["docRun"] == nil, message["transcript"] == nil,
        message["run"]?["model"]?.string == model,
        (message["toolCalls"]?.array ?? []).isEmpty, (message["webSearches"]?.array ?? []).isEmpty,
        let prompt = message["usage"]?["promptTokens"]?.integer,
        (1...1_000_000_000).contains(prompt),
        let count = message["run"]?["nativeContext"]?["count"]?.integer,
        let key = message["run"]?["nativeContext"]?["key"]?.string,
        prefixKey(body, count: count) == key
      else { break }
      let answer = min(1_000_000_000, max(0, message["usage"]?["completionTokens"]?.integer ?? 0))
      let extra = max(0, answer + 4 - messageTokens(message))
      return (
        prompt,
        NativeReplyBudget.estimate(input: Array(input.dropFirst(count)), instructions: "") + extra
      )
    }
    return (0, fallback)
  }
}
