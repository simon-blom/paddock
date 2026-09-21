import Foundation

/// Item identities, not event counts: speech-start and commit may both name
/// the same utterance, and an older final result may arrive after the next
/// speech-start. Stop waits for the ordered input barrier and every item.
public struct NativeSpeechProgress: Sendable {
  public private(set) var pending = Set<String>()
  public private(set) var speaking: String?
  public private(set) var draining = false
  private var completed = Set<String>()
  public init() {}
  public var waiting: Bool { draining || !pending.isEmpty }
  public mutating func beginDrain() { draining = true }
  public mutating func fail() {
    pending.removeAll()
    speaking = nil
    draining = false
  }
  /// False means a duplicate completion, which must not append words twice.
  @discardableResult public mutating func receive(_ event: [String: ConversationValue]) -> Bool {
    let id = event["item_id"]?.string ?? "legacy-item"
    switch event["type"]?.string {
    case "input_audio_buffer.speech_started":
      speaking = id
      if !completed.contains(id) { pending.insert(id) }
    case "input_audio_buffer.committed":
      if !completed.contains(id) { pending.insert(id) }
    case "input_audio_buffer.speech_stopped":
      if speaking == id { speaking = nil }
    case "input_audio_buffer.drained":
      draining = false
      speaking = nil
      for value in event["pending_item_ids"]?.array ?? [] {
        if let id = value.string, !completed.contains(id) { pending.insert(id) }
      }
    case "conversation.item.input_audio_transcription.completed":
      pending.remove(id)
      if speaking == id { speaking = nil }
      if id == "legacy-item" { return true }
      return completed.insert(id).inserted
    default: break
    }
    return true
  }
}
