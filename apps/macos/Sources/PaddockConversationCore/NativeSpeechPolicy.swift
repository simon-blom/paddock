import Foundation

/// Match Studio's capability-gated realtime request. Older runners treat
/// `paddock_verbose` as a request for final-pass Whisper word alignment, not
/// just more JSON. Asking unconditionally can fail only at the final word.
public enum NativeSpeechPolicy {
  public static let sampleRate = 16000
  public static let maximumUtteranceSeconds = 30.0
  public static let finalizationSeconds = 60.0
  // PCM16 mono, bounded by the attachment service's 100 MiB limit.
  public static let recordingSeconds = Double(100 * 1024 * 1024 - 44) / 32000
  public static func liveSupported(_ capabilities: [String: ConversationValue]) -> Bool {
    capabilities["realtime_transcription"]?["supported"]?.bool != false
  }
  public static func recordingLimit(capabilities: [[String: ConversationValue]], live: Bool)
    -> Double
  {
    guard !live else { return recordingSeconds }
    return capabilities.compactMap { $0["transcription_max_clip_s"]?.double }
      .filter { $0.isFinite && $0 > 0 }.reduce(recordingSeconds, min)
  }
  public static func utteranceLimit(_ capabilities: [[String: ConversationValue]]) -> Double {
    capabilities.compactMap {
      $0["realtime_transcription"]?["utterance_max_s"]?.double
        ?? $0["transcription_max_clip_s"]?.double
    }
    .filter { $0.isFinite && $0 > 0 }.reduce(maximumUtteranceSeconds, min)
  }
  public static func transcription(
    capabilities: [String: ConversationValue],
    dictation: Bool, language: String?
  ) -> [String: ConversationValue] {
    var result: [String: ConversationValue] = [:]
    let times = capabilities["timestamp_granularities"]?.array ?? []
    let live = capabilities["realtime_transcription"]?.object
    if live?["final_revision"]?.bool == true { result["paddock_final_revision"] = .bool(true) }
    let enrich =
      live.map { $0["supported"]?.bool == true && $0["enrichment"]?.bool == true }
      ?? (times.contains(.string("segment")) && times.contains(.string("word")))
    if !dictation, enrich {
      result["paddock_verbose"] = .bool(true)
    }
    if let language, !language.isEmpty, language != "auto" {
      result["language"] = .string(language)
    }
    return result
  }

  /// File policy is deliberately different from live policy. An explicit task
  /// instruction wins over Granite Plus's implicit word-timing instruction.
  public static func fileFields(
    capabilities: [String: ConversationValue], cloud: Bool,
    instruction: String?, language: String?
  ) -> [(String, String)] {
    var fields = [("response_format", cloud ? "json" : "verbose_json")]
    if !cloud { fields.append(("stream", "true")) }
    let prompt = instruction?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    for unit in capabilities["timestamp_granularities"]?.array?.compactMap(\.string) ?? [] {
      if unit == "segment" || (unit == "word" && prompt.isEmpty) {
        fields.append(("timestamp_granularities[]", unit))
      }
    }
    if capabilities["include"]?.array?.contains(.string("logprobs")) == true {
      fields.append(("include[]", "logprobs"))
    }
    if !prompt.isEmpty { fields.append(("prompt", prompt)) }
    if let language, !language.isEmpty, language != "auto" { fields.append(("language", language)) }
    return fields
  }
}
