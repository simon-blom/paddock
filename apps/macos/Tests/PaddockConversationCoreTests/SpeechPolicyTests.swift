import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Native speech capability policy")
struct SpeechPolicyTests {
  @Test func wordRenderingMatchesSharedWebFixtures() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "speech-rendering", withExtension: "json", subdirectory: "Fixtures"))
    let cases = try JSONDecoder().decode(
      [[String: ConversationValue]].self, from: Data(contentsOf: url))
    for row in cases {
      let actual = NativeSpeechMetadata.renderWords(
        row["meta"]?.object ?? [:], text: row["text"]?.string ?? "",
        streaming: row["streaming"]?.bool ?? false)
      #expect(actual == row["expected"]?.array, "\(row["name"]?.string ?? "")")
    }
  }
  @Test func finalRevisionIsNegotiatedForBothLiveAndDictation() {
    for dictate in [false, true] {
      let caps: [String: ConversationValue] = [
        "realtime_transcription": .object([
          "supported": .bool(true), "enrichment": .bool(false), "final_revision": .bool(true),
        ])
      ]
      let fields = NativeSpeechPolicy.transcription(
        capabilities: caps, dictation: dictate, language: "sv")
      #expect(fields["paddock_final_revision"]?.bool == true)
      #expect(fields["paddock_verbose"] == nil)
    }
  }
  @Test func matchesWebStopAndDelayedCompletionSequences() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "speech-progress", withExtension: "json", subdirectory: "Fixtures"))
    let fixtures = try JSONDecoder().decode(
      [[String: ConversationValue]].self, from: Data(contentsOf: url))
    for fixture in fixtures {
      var progress = NativeSpeechProgress()
      for step in fixture["steps"]?.array ?? [] {
        if step["drain"]?.bool == true { progress.beginDrain() }
        if let event = step["event"]?.object {
          #expect(progress.receive(event) == (step["accepted"]?.bool ?? true))
        }
        #expect(progress.pending.count == step["pending"]?.integer)
        #expect(progress.waiting == step["waiting"]?.bool)
        #expect(progress.speaking == step["speaking"]?.string)
      }
    }
    var failed = NativeSpeechProgress()
    failed.beginDrain()
    failed.fail()
    #expect(!failed.waiting)
  }
  @Test func matchesTheSameCapabilityFixturesAsWebStudio() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "speech-capabilities", withExtension: "json", subdirectory: "Fixtures"))
    let cases = try JSONDecoder().decode(
      [[String: ConversationValue]].self, from: Data(contentsOf: url))
    for row in cases {
      let caps = try #require(row["caps"]?.object)
      let live = NativeSpeechPolicy.transcription(
        capabilities: caps, dictation: false, language: "sv")
      #expect(
        (live["paddock_verbose"]?.bool ?? false) == row["enrich"]?.bool,
        "\(row["name"]?.string ?? "")")
      #expect(live["language"]?.string == "sv")
      #expect(
        NativeSpeechPolicy.transcription(capabilities: caps, dictation: true, language: "sv")[
          "paddock_verbose"] == nil)
    }
  }
  @Test func instructionsWinOverGraniteWordTimingAndLimitsMatchTheModel() {
    typealias V = ConversationValue
    let caps: [String: V] = [
      "timestamp_granularities": .array([.string("word")]),
      "include": .array([.string("logprobs")]), "transcription_max_clip_s": .number(120),
    ]
    let fields = NativeSpeechPolicy.fileFields(
      capabilities: caps, cloud: false, instruction: "  Identify speakers  ", language: "sv")
    #expect(fields.contains { $0.0 == "prompt" && $0.1 == "Identify speakers" })
    #expect(!fields.contains { $0.0 == "timestamp_granularities[]" })
    #expect(fields.contains { $0.0 == "include[]" && $0.1 == "logprobs" })
    #expect(
      NativeSpeechPolicy.fileFields(
        capabilities: caps, cloud: false, instruction: nil, language: nil
      ).contains { $0.0 == "timestamp_granularities[]" && $0.1 == "word" })
    #expect(NativeSpeechPolicy.recordingLimit(capabilities: [caps], live: false) == 120)
    #expect(NativeSpeechPolicy.recordingLimit(capabilities: [caps], live: true) > 120)
    #expect(NativeSpeechPolicy.utteranceLimit([caps]) == 30)
    #expect(
      NativeSpeechPolicy.utteranceLimit([
        ["realtime_transcription": .object(["utterance_max_s": .number(8)])]
      ]) == 8)
  }
  @Test func alignmentMergeIsStrictUnicodeAwareAndPreservesConfidence() {
    typealias V = ConversationValue
    let original: [String: V] = [
      "words": .array([
        .object(["word": .string("Hej,"), "confidence": .number(0.9)]),
        .object(["word": .string("你好！")]),
      ])
    ]
    let spans: [V] = [
      .object(["word": .string("hej"), "start": .number(0.1), "end": .number(0.5)]),
      .object(["word": .string("你"), "start": .number(0.6), "end": .number(1)]),
      .object(["word": .string("好"), "start": .number(1), "end": .number(1.3)]),
    ]
    let merged = NativeSpeechAlignment.merge(meta: original, text: "", aligned: spans, duration: 2)
    #expect(merged?.first?["word"]?.string == "Hej,")
    #expect(merged?.first?["confidence"] == .number(0.9))
    #expect(merged?.last?["start"] == .number(0.6))
    #expect(merged?.last?["end"] == .number(1.3))
    #expect(
      NativeSpeechAlignment.merge(
        meta: original, text: "", aligned: Array(spans.dropLast()), duration: 2) == nil)
    #expect(
      NativeSpeechAlignment.merge(meta: original, text: "", aligned: spans, duration: 1) == nil)
    #expect(
      NativeSpeechAlignment.merge(meta: [:], text: "Wrong words", aligned: spans, duration: 2)
        == nil)
    #expect(
      NativeSpeechAlignment.merge(
        meta: original, text: "", aligned: Array(spans.reversed()), duration: 2) == nil)
  }
  @Test func metalWhisperDoesNotRequestUnavailableFinalAlignment() {
    let config = NativeSpeechPolicy.transcription(
      capabilities: ["timestamp_granularities": .array([.string("segment")])], dictation: false,
      language: "sv")
    #expect(config == ["language": .string("sv")])
  }
  @Test func enrichedWhisperOnlyWhenBothGranularitiesAreAdvertised() {
    let capabilities: [String: ConversationValue] = [
      "timestamp_granularities": .array([.string("segment"), .string("word")])
    ]
    #expect(
      NativeSpeechPolicy.transcription(
        capabilities: capabilities, dictation: false, language: "auto") == [
          "paddock_verbose": .bool(true)
        ])
    #expect(
      NativeSpeechPolicy.transcription(capabilities: capabilities, dictation: true, language: "sv")
        == ["language": .string("sv")])
  }
  @Test func generativeAndUnknownCapabilitiesDoNotOptIn() {
    for capabilities: [String: ConversationValue] in [
      [:], ["timestamp_granularities": .array([])],
      ["timestamp_granularities": .array([.string("word")])],
    ] {
      #expect(
        NativeSpeechPolicy.transcription(
          capabilities: capabilities, dictation: false, language: nil
        ).isEmpty)
    }
  }
  @Test func liveWordTimesUseRecordingOffsetsAndRetainConfidence() throws {
    typealias V = ConversationValue
    let utterance: [String: V] = [
      "transcript": .string("Testar funktionen"), "paddock_audio_start_ms": .number(1200),
      "usage": .object(["seconds": .number(2)]),
      "paddock_verbose": .object([
        "language": .string("sv"), "duration": .number(2),
        "segments": .array([
          .object([
            "text": .string("Testar funktionen"), "start": .number(0.1), "end": .number(1.8),
          ])
        ]),
        "words": .array([
          .object([
            "word": .string("Testar"), "start": .number(0.1), "end": .number(0.7),
            "paddock_confidence": .number(0.98),
          ]),
          .object([
            "word": .string("funktionen"), "start": .number(0.7), "end": .number(1.8),
            "paddock_confidence": .number(0.95),
          ]),
        ]),
      ]),
    ]
    let first = NativeSpeechMetadata.appendingLive(utterance, to: [:])
    #expect(first["words"]?.array?.first?["start"]?.double == 1.3)
    #expect(first["words"]?.array?.last?["end"]?.double == 3)
    #expect(abs((first["words"]?.array?.first?["confidence"]?.double ?? 0) - 0.98) < 1e-12)
    var next = utterance
    next["paddock_audio_start_ms"] = .number(5000)
    let both = NativeSpeechMetadata.appendingLive(next, to: first)
    #expect(both["words"]?.array?.count == 4)
    #expect(both["words"]?.array?.last?["segment"]?.integer == 1)
    #expect(both["words"]?.array?.last?["end"]?.double == 6.8)
    #expect(both["durationS"]?.double == 7)
    #expect(both["language"]?.string == "sv")
  }
}
