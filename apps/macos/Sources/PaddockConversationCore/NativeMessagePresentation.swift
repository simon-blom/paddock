import Foundation

/// Native parity with studio/src/lib/message-presentation.ts. Projection is an
/// allowlist, never a dump of provider responses, tool headers or credentials.
enum NativeMessagePresentation {
  typealias V = ConversationValue
  typealias O = [String: V]
  static func duration(_ ms: Double) -> String {
    guard ms.isFinite, ms >= 0 else { return "" }
    if ms < 1000 { return "\(Int(ms.rounded()))ms" }
    return ms < 10000 ? String(format: "%.1fs", ms / 1000) : "\(Int((ms / 1000).rounded()))s"
  }
  static func cost(_ usd: Double) -> String {
    guard usd.isFinite, usd >= 0 else { return "" }
    return usd < 0.0005 ? "<$0.001" : String(format: usd >= 0.1 ? "$%.2f" : "$%.3f", usd)
  }
  static func speed(_ n: Double?) -> String? {
    n.flatMap { $0 > 0 && $0.isFinite ? "\(Int($0.rounded())) tok/s" : nil }
  }
  static func join(_ parts: [String?]) -> String {
    parts.compactMap { $0 }.filter { !$0.isEmpty }.joined(separator: " · ")
  }
  static func measuredSpeed(_ u: O) -> String? {
    speed(u["tps"]?.double).map {
      $0 + (u["timingSource"]?.string == "engine" ? " decode" : " end-to-end")
    }
  }
  static func footer(_ raw: O, realtime: Double? = nil) -> String {
    let u = NativeResponseMetrics.presentation(raw)
    let outputTokens = (u["completionTokens"]?.integer ?? 0) + (u["reasoningTokens"]?.integer ?? 0)
    guard outputTokens > 0 || u["costUsd"]?.double != nil || realtime != nil else { return "" }
    var parts: [String?] = []
    if outputTokens > 0 { parts.append("\(outputTokens) tokens") }
    if realtime == nil { parts.append(speed(u["tps"]?.double)) }
    if let ms = u["ms"]?.double ?? u["answerMs"]?.double, ms > 0 {
      let rate = realtime.map { $0 >= 10 ? String(Int($0.rounded())) : String(format: "%.1f", $0) }
      parts.append(duration(ms) + (rate.map { " (\($0)× realtime)" } ?? ""))
    }
    parts.append(u["costUsd"]?.double.map(cost))
    return join(parts)
  }
  static func hint(_ raw: O) -> String {
    let u = NativeResponseMetrics.presentation(raw)
    guard let ms = u["ms"]?.double, ms > 0 else { return "" }
    return join([
      "Total time from send to done",
      u["ttftMs"]?.double.map { "\(duration($0)) to the first token" },
      u["reasoningMs"]?.double.map { "\(duration($0)) thinking" },
      u["answerMs"]?.double.map { "\(duration($0)) writing the answer" },
      u["provider"]?.string.map { "served by \($0)" }, measuredSpeed(u),
    ])
  }
  static func sections(run: O?, usage raw: O) -> [V] {
    let u = NativeResponseMetrics.presentation(raw)
    var result: [V] = []
    func section(_ id: String, _ title: String, _ rows: [(String, String)]) {
      result.append(
        .object([
          "id": .string(id), "title": .string(title),
          "rows": .array(
            rows.map {
              .object(["label": .string($0.0), "value": .string($0.1.isEmpty ? "-" : $0.1)])
            }),
        ]))
    }
    if let r = run {
      let p = r["params"]?.object ?? [:]
      func dial(_ key: String) -> String {
        guard let n = p[key]?.double else { return "default" }
        return n.rounded() == n ? String(format: "%.1f", n) : String(format: "%g", n)
      }
      var sampling = [
        "temp \(dial("temperature"))", "top-p \(dial("topP"))",
        "top-k \(p["topK"]?.integer.map { $0 == 0 ? "off" : String($0) } ?? "default")",
      ]
      for (key, label) in [
        ("minP", "min-p"), ("presencePenalty", "pres"), ("frequencyPenalty", "freq"),
        ("repeatPenalty", "repeat"),
      ] {
        if let n = p[key]?.double, n != 0, key != "repeatPenalty" || n != 1 {
          sampling.append("\(label) \(key == "minP" ? String(format: "%g", n) : dial(key))")
        }
      }
      var rows = [("Model", r["model"]?.string ?? "-")]
      if let spec = r["spec"]?.string, !spec.isEmpty { rows.append(("Speculation", spec)) }
      rows += [
        (
          "System prompt",
          r["systemPromptName"]?.string
            ?? ((r["systemPrompt"]?.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
              .isEmpty ? "None" : "Custom")
        ),
        ("Sampling", join(sampling)),
        (
          "Reasoning",
          p["thinking"]?.bool == false
            ? "off"
            : p["reasoningEffort"]?.string.flatMap { $0.isEmpty ? nil : $0 } ?? "Model default"
        ),
        (
          "Max tokens", (p["maxTokens"]?.integer ?? r["maxTokens"]?.integer).map(String.init) ?? "-"
        ),
      ]
      if let seed = p["seed"]?.integer { rows[rows.count - 1].1 += " · seed \(seed)" }
      if let tools = r["tools"]?.array?.compactMap(\.string), !tools.isEmpty {
        rows.append(("Tools", tools.joined(separator: ", ")))
      } else if r["tools"] == nil {
        rows.append(("Tools", "Not recorded"))
      }
      if r["contended"]?.bool == true {
        rows.append(("Concurrency", "Other compare lanes shared the GPU during this run"))
      }
      section("provenance", "Provenance", rows)
    }
    if !u.isEmpty {
      var rows = [
        (
          "Tokens",
          join([
            "\(u["promptTokens"]?.integer.map(String.init) ?? "-") in",
            "\(u["completionTokens"]?.integer.map { String($0 + (u["reasoningTokens"]?.integer ?? 0)) } ?? "-") out",
            u["reasoningTokens"]?.integer.flatMap { $0 > 0 ? "\($0) reasoning" : nil },
          ])
        ),
        (
          "Speed",
          join([
            measuredSpeed(u), u["ttftMs"]?.double.map { "TTFT \(duration($0))" },
            u["ms"]?.double.map { "\(duration($0)) total" },
          ])
        ),
      ]
      if u["reasoningMs"]?.double != nil || u["answerMs"]?.double != nil {
        rows.append(
          (
            "Phases",
            join([
              u["reasoningMs"]?.double.map { "\(duration($0)) thinking" },
              u["answerMs"]?.double.map { "\(duration($0)) writing" },
            ])
          ))
      }
      if u["timingSource"]?.string == "engine" {
        rows.append(
          (
            "Engine timing",
            join([
              u["queueMs"]?.double.map { "\(duration($0)) queued" },
              u["prefillMs"]?.double.map { "\(duration($0)) prefill" },
              u["decodeMs"]?.double.map { "\(duration($0)) decode (all rounds)" },
            ])
          ))
      }
      if let provider = u["provider"]?.string { rows.append(("Served by", provider)) }
      if let usd = u["costUsd"]?.double { rows.append(("Cost", cost(usd))) }
      section("metrics", "Metrics", rows)
    }
    if let g = run?["gpu"]?.object {
      var rows: [(String, String)] = []
      if let name = g["device"]?.string { rows.append(("Device", name)) }
      rows.append(
        (
          "Load",
          join([
            g["utilPeak"]?.double.map { "\(Int($0))% util" },
            g["memUsedPeak"]?.double.map {
              String(
                format: $0 / pow(1024, 3) >= 10 ? "%.0f GB VRAM" : "%.1f GB VRAM", $0 / pow(1024, 3)
              )
            }, g["powerPeakW"]?.double.map { "\(Int($0.rounded())) W" },
            g["tempPeakC"]?.double.map { "\(Int($0))°C" },
          ])
        ))
      if g["batchPeak"] != nil || g["kvTotal"] != nil || g["tokSPeak"] != nil {
        rows.append(
          (
            "Engine",
            join([
              g["batchPeak"]?.integer.map { "batch \($0)" },
              g["kvTotal"]?.integer.map {
                "KV \(g["kvPeak"]?.integer.map(String.init) ?? "-")/\($0)"
              }, speed(g["tokSPeak"]?.double).map { "\($0) peak" },
            ])
          ))
      }
      section("gpu", "GPU environment (peak)", rows)
    }
    return result
  }
}

extension NativeStudioRuntime {
  static func recordedSpec(_ run: O?) -> String {
    let value = (run?["spec"]?.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    return ["off", "false", "no", "none", "0", "disabled"].contains(value.lowercased()) ? "" : value
  }
  static func tokenLimitNote(_ usage: O) -> String {
    guard let reasoning = usage["reasoningTokens"]?.integer, reasoning > 0,
      let answer = usage["completionTokens"]?.integer, answer > 0
    else {
      return
        "Reply reached its output limit. The context window also includes your prompt and history."
    }
    return
      "Reply reached its output limit after \(reasoning + answer) generated tokens, including \(reasoning) thinking tokens. The context window also includes your prompt and history."
  }
  func runSnapshot(modelID: String, body: O) -> O {
    let model = models.first { $0["id"]?.string == modelID }
    var params = document?.fields["params"]?.object ?? [:]
    params["maxTokens"] = body["max_output_tokens"] ?? .null
    let tools = (body["tools"]?.array ?? []).filter { $0["type"]?.string == "mcp" }.flatMap {
      tool -> [V] in
      guard let label = tool["server_label"]?.string else { return [] }
      if let names = tool["allowed_tools"]?.array?.compactMap(\.string), !names.isEmpty {
        return names.map { .string("\(label):\($0)") }
      }
      return [.string(label)]
    }
    let shared =
      selected.filter { id in models.contains { $0["id"]?.string == id && $0["port"] != nil } }
      .count > 1
    return [
      "model": .string(modelID), "modelName": model?["title"] ?? .string(modelID),
      "vendor": model?["vendor"] ?? .string(""),
      "systemPrompt": document?.fields["systemPrompt"] ?? .string(""),
      "params": .object(params), "spec": model?["spec"] ?? .string(""), "tools": .array(tools),
      "contended": .bool(shared && model?["port"] != nil), "at": Self.now,
    ]
  }
}
