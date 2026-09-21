import Foundation
import Testing

@testable import PaddockConversationCore

struct ResponsePresentationTests {
  typealias V = ConversationValue
  typealias O = [String: V]

  @Test func actualArtifactTurnNeverDividesToolTokensByFinalAnswerTime() throws {
    let legacy: O = [
      "promptTokens": .number(2719), "completionTokens": .number(1386),
      "reasoningTokens": .number(99),
      "ms": .number(56898.261834), "answerMs": .number(3464.010292),
      "reasoningMs": .number(49009.056542), "tps": .number(400.11428),
    ]
    let repaired = NativeResponseMetrics.presentation(legacy)
    #expect(abs(try #require(repaired["tps"]?.double) - 26.09921555) < 0.000001)
    #expect(repaired["answerMs"] == nil && repaired["reasoningMs"] == nil)
    var metrics = NativeResponseMetrics()
    metrics.observe(delta("output_text"), seconds: 53.434)
    let response: O = [
      "usage": .object([
        "output_tokens": .number(1485),
        "output_tokens_details": .object(["reasoning_tokens": .number(99)]),
      ]),
      "paddock_timing": .object([
        "source": .string("engine"), "version": .number(1), "decode_ms": .number(50000),
        "prefill_ms": .number(4000), "queue_ms": .number(10),
      ]),
    ]
    let measured = try #require(metrics.usage(response, seconds: 56.898))
    #expect(measured["tps"]?.double == 29.7)
    #expect(measured["timingSource"]?.string == "engine")
    #expect(
      metrics.usage(response, seconds: 56.898, previous: legacy)?["timingSource"]?.string
        == "end-to-end")
    #expect(
      metrics.usage(response, seconds: 56.898, previous: measured)?["decodeMs"]?.integer == 100000)
  }

  @Test func sharesFullPresentationFixturesWithActualWebImplementation() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "message-presentation", withExtension: "json", subdirectory: "Fixtures"))
    let cases = try JSONDecoder().decode([O].self, from: Data(contentsOf: url))
    for row in cases {
      let usage = row["usage"]?.object ?? [:]
      let sections = NativeMessagePresentation.sections(run: row["run"]?.object, usage: usage)
      #expect(
        NativeMessagePresentation.footer(usage, realtime: row["realtime"]?.double)
          == row["footer"]?.string)
      #expect(NativeMessagePresentation.hint(usage) == row["hint"]?.string)
      #expect(sections == row["sections"]?.array, "\(row["name"]?.string ?? "")")
      let encoded = String(decoding: try JSONEncoder().encode(sections), as: UTF8.self)
      #expect(!encoded.contains("NEVER_PROJECT"))
      #expect(!encoded.contains("System text"))
    }
  }

  func delta(_ type: String, _ text: String = "token") -> O {
    ["type": .string("response.\(type).delta"), "delta": .string(text)]
  }
  var terminal: O {
    [
      "provider": .string("Example host"),
      "usage": .object([
        "input_tokens": .number(80), "output_tokens": .number(320),
        "output_tokens_details": .object(["reasoning_tokens": .number(200)]),
        "cost": .number(0.012),
      ]),
    ]
  }

  @Test func ignoresDiscoveryAndArgumentDeltasAndMeasuresActualTokenPhases() throws {
    var metrics = NativeResponseMetrics()
    metrics.observe(delta("mcp_call_arguments"), seconds: 0.1)
    metrics.observe(delta("output_text", ""), seconds: 0.2)
    #expect(metrics.firstToken == nil)
    metrics.observe(delta("reasoning_summary_text"), seconds: 0.6)
    metrics.observe(delta("output_text"), seconds: 3.6)
    metrics.observe(["type": .string("response.completed")], seconds: 5.6)
    let usage = try #require(metrics.usage(terminal, seconds: 7.6))
    #expect(
      usage["ms"]?.double == 5600,
      "Saving/checkpointing after the terminal frame must not reduce measured speed")
    #expect(usage["completionTokens"]?.integer == 120)
    #expect(usage["reasoningTokens"]?.integer == 200)
    #expect(usage["ttftMs"]?.integer == 600)
    #expect(abs(try #require(usage["tps"]?.double) - 320.0 / 5.6) < 0.001)
    #expect(usage["reasoningTps"] == nil)
    #expect(usage["answerMs"] == nil)
    #expect(usage["costUsd"]?.double == 0.012)
    #expect(usage["provider"]?.string == "Example host")
  }

  @Test func noTerminalUsageNeverCreatesFakeCountsOrCost() {
    var metrics = NativeResponseMetrics()
    metrics.observe(delta("output_text"), seconds: 1)
    #expect(metrics.usage(nil, seconds: 2) == nil)
    #expect(metrics.usage(["error": .string("Provider rejected request")], seconds: 2) == nil)
    let usage = metrics.usage(["usage": .object(["output_tokens": .number(5)])], seconds: 2)
    #expect(usage?["promptTokens"] == nil)
    #expect(usage?["costUsd"] == nil)
    #expect(usage?["reasoningTokens"] == nil)
    #expect(usage?["tps"]?.double == 2.5)
  }

  @Test func continueAccumulatesCountsPhasesAndCostWithoutFabricatingOldSpeed() throws {
    var metrics = NativeResponseMetrics()
    metrics.observe(delta("reasoning"), seconds: 0.6)
    metrics.observe(delta("output_text"), seconds: 3.6)
    let first = try #require(metrics.usage(terminal, seconds: 5.6))
    let continued = try #require(metrics.usage(terminal, seconds: 5.6, previous: first))
    #expect(continued["completionTokens"]?.integer == 240)
    #expect(continued["reasoningTokens"]?.integer == 400)
    #expect(continued["ms"]?.double == 11200)
    #expect(continued["costUsd"]?.double == 0.024)
    #expect(abs(try #require(continued["tps"]?.double) - 320.0 / 5.6) < 0.001)
    #expect(metrics.usage(nil, seconds: 2, previous: first) == first)
    let old: O = [
      "completionTokens": .number(109), "reasoningTokens": .number(126), "ms": .number(12245),
    ]
    let missing = metrics.usage(terminal, seconds: 5.6, previous: old)
    #expect(missing?["timingSource"]?.string == "end-to-end")
    #expect(abs((missing?["tps"]?.double ?? 0) - 555.0 / 17.845) < 0.001)
    #expect(missing?["reasoningTps"] == nil)
    #expect(missing?["answerMs"] == nil)
    #expect(missing?["reasoningMs"] == nil)
  }

  @Test func legacyQwenDetailsDoNotInventPhaseSpeed() {
    let usage: O = [
      "promptTokens": .number(2722), "completionTokens": .number(109),
      "reasoningTokens": .number(126), "ms": .number(12245.7), "ttftMs": .number(4443.87),
    ]
    #expect(NativeMessagePresentation.footer(usage) == "235 tokens · 19 tok/s · 12s")
    let details = NativeMessagePresentation.sections(
      run: ["model": .string("qwen"), "maxTokens": .number(4096), "params": .object([:])],
      usage: usage)
    let rows = details.flatMap { $0["rows"]?.array ?? [] }
    #expect(rows.contains { $0["label"]?.string == "Max tokens" && $0["value"]?.string == "4096" })
    #expect(!rows.contains { $0["label"]?.string == "Phases" })
    #expect(NativeStudioRuntime.recordedSpec(["spec": .string("off")]).isEmpty)
    #expect(
      NativeStudioRuntime.tokenLimitNote(usage)
        == "Reply reached its output limit after 235 generated tokens, including 126 thinking tokens. The context window also includes your prompt and history."
    )
  }

  @Test func discoveryIsNotACallAndRealSearchAndArtifactCallsSurvive() {
    let discovery = V.object([
      "id": .string("discovery"), "type": .string("mcp_list_tools"), "name": .string("artifacts"),
    ])
    #expect(!NativeStudioRuntime.visibleToolCall(discovery))
    for type in ["mcp_call", "mcp_approval_request"] {
      #expect(
        NativeStudioRuntime.visibleToolCall(
          .object(["type": .string(type), "name": .string("mcp_search_tools")])))
    }
    #expect(NativeStudioRuntime.visibleToolCall(.object(["name": .string("legacy_tool")])))
    let artifact = V.object([
      "name": .string("artifacts__artifact_create"),
      "output": .string("{\"id\":\"art_123456789abc\"}"),
    ])
    #expect(NativeStudioRuntime.toolArtifactID(artifact) == "art_123456789abc")
    #expect(
      NativeStudioRuntime.toolArtifactID(
        .object(["name": .string("mcp_search_tools"), "output": .string("art_123456789abc")]))
        == nil)
  }
}
