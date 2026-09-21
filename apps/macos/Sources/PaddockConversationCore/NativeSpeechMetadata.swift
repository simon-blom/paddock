import Foundation

/// The same transcript schema for uploaded files and live utterances. Live
/// times are utterance-local on the wire, but recording-relative in storage.
public enum NativeSpeechMetadata {
  public typealias V = ConversationValue
  public typealias O = [String: V]
  /// The web Studio's renderWords contract. Streaming text is authoritative:
  /// completed-utterance metadata must never hide the next live utterance.
  /// Sentence timing is a seek target, not a made-up clock for every word.
  public static func renderWords(_ meta: O, text: String, streaming: Bool) -> [V] {
    func plain(_ text: String, segment: Int = -1, start: V? = nil) -> [V] {
      text.split(whereSeparator: \.isWhitespace).map {
        var row: O = ["word": .string(String($0)), "segment": .number(Decimal(segment))]
        row["start"] = start
        return .object(row)
      }
    }
    if streaming { return plain(text) }
    let segments = meta["segments"]?.array ?? []
    if let words = meta["words"]?.array, !words.isEmpty {
      return words.map { value in
        var row = value.object ?? [:]
        let segment =
          row["start"]?.double.flatMap { time in
            segments.lastIndex { ($0["start"]?.double ?? .infinity) <= time + 0.001 }
          } ?? -1
        row["segment"] = .number(Decimal(segment))
        return .object(row)
      }
    }
    if !segments.isEmpty {
      return segments.enumerated().flatMap { index, segment -> [V] in
        if let words = segment["words"]?.array, !words.isEmpty {
          return words.map { value in
            var row = value.object ?? [:]
            row["segment"] = .number(Decimal(index))
            row["start"] = segment["start"]
            row["end"] = nil
            return .object(row)
          }
        }
        return plain(segment["text"]?.string ?? "", segment: index, start: segment["start"])
      }
    }
    return plain(text)
  }
  public static func word(_ value: V) -> V {
    var row = value.object ?? [:]
    row["confidence"] = row["confidence"] ?? row["paddock_confidence"]
    row["alt"] = row["alt"] ?? row["paddock_alt"]
    row["margin"] = row["margin"] ?? row["paddock_margin"]
    return .object(row)
  }
  public static func file(_ response: O) -> O {
    var meta = response["paddock_verbose"]?.object ?? response
    meta["durationS"] = meta["durationS"] ?? meta["duration"]
    meta["guards"] =
      response["paddock_guards"] ?? meta["paddock_guards"] ?? meta["guards"] ?? .array([])
    var segments = meta["segments"]?.array ?? []
    for i in segments.indices {
      var segment = segments[i].object ?? [:]
      segment["words"] = segment["words"]?.array.map { .array($0.map(word)) }
      segments[i] = .object(segment)
    }
    meta["segments"] = .array(segments)
    meta["words"] = .array(
      (meta["words"]?.array ?? meta["paddock_words"]?.array ?? []).map { value in
        var w = word(value).object ?? [:]
        if w["segment"] == nil, let start = w["start"]?.double {
          w["segment"] = .number(
            Decimal(segments.lastIndex { ($0["start"]?.double ?? .infinity) <= start } ?? 0))
        }
        return .object(w)
      })
    return meta
  }
  public static func appendingLive(_ response: O, to previous: O) -> O {
    var result = previous
    let offset = max(0, response["paddock_audio_start_ms"]?.double ?? 0) / 1000
    let local = file(response)
    let duration = local["durationS"]?.double ?? response["usage"]?["seconds"]?.double ?? 0
    var segments = result["segments"]?.array ?? []
    var words = result["words"]?.array ?? []
    var guards = result["guards"]?.array ?? []
    let base = segments.count
    func shift(_ value: V) -> V {
      var row = value.object ?? [:]
      for key in ["start", "end"] {
        if let time = row[key]?.double, time.isFinite, time >= 0 {
          row[key] = .number(Decimal(offset + time))
        }
      }
      return .object(row)
    }
    func shiftWord(_ value: V) -> V {
      var row = shift(word(value)).object ?? [:]
      row["segment"] = .number(Decimal(base + (row["segment"]?.integer ?? 0)))
      return .object(row)
    }
    for value in local["segments"]?.array ?? [] {
      var segment = shift(value).object ?? [:]
      segment["words"] = segment["words"]?.array.map { .array($0.map(shiftWord)) }
      segments.append(.object(segment))
    }
    if segments.count == base, let text = response["transcript"]?.string, !text.isEmpty {
      segments.append(
        .object([
          "text": .string(text), "start": .number(Decimal(offset)),
          "end": .number(Decimal(offset + duration)),
        ]))
    }
    words += (local["words"]?.array ?? []).map(shiftWord)
    for value in local["guards"]?.array ?? [] {
      var guardRow = shift(value).object ?? [:]
      guardRow["start"] = guardRow["start"] ?? .number(Decimal(offset))
      guardRow["end"] = guardRow["end"] ?? .number(Decimal(offset + duration))
      guards.append(.object(guardRow))
    }
    result["segments"] = .array(segments)
    result["words"] = .array(words)
    result["guards"] = .array(guards)
    result["durationS"] = .number(Decimal(max(result["durationS"]?.double ?? 0, offset + duration)))
    if let language = local["language"]?.string { result["language"] = .string(language) }
    return result
  }
}
