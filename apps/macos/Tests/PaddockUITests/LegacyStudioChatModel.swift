import Foundation
import Observation
import PaddockClient
import PaddockTranscript

/// App lifetime, not view lifetime. Switching tabs/windows cannot orphan a
/// generation. Rust owns the durable conversation; this is its UI projection.
@MainActor @Observable
public final class StudioChatModel {
  public private(set) var conversation: ChatDocument?
  public private(set) var history: [ChatSummary] = []
  public private(set) var busy = false
  public private(set) var error: String?
  public var selectedRunnerID: String?
  @ObservationIgnored private var transcriptStorage: TranscriptSession?
  public var transcript: TranscriptSession {
    if let transcriptStorage { return transcriptStorage }
    let value = TranscriptSession()
    transcriptStorage = value
    return value
  }
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var streamID: String?
  @ObservationIgnored private var generation: UInt64 = 0
  @ObservationIgnored private var stopped = false
  @ObservationIgnored private var cancelRequested = false

  public init(client: any ManagerLoading) { self.client = client }
  // Isolated integration tests supply generated resources, never user data.
  init(client: any ManagerLoading, transcript: TranscriptSession) {
    self.client = client
    transcriptStorage = transcript
  }
  public func refreshHistory() async {
    do {
      let result = try await client.chat(.list)
      if !stopped { history = result.conversations ?? [] }
    } catch { if !stopped { self.error = error.localizedDescription } }
  }
  public func newChat() async {
    // Initial slice owns one active conversation. Explicitly stop first,
    // rather than silently cancelling or misrouting a response during navigation.
    guard !busy else {
      error = "Stop the current response before starting another chat."
      return
    }
    generation &+= 1
    conversation = nil
    error = nil
    await transcriptStorage?.show(nil)
  }
  public func open(_ id: String) async {
    guard !busy else {
      error = "Stop the current response before opening another chat."
      return
    }
    generation &+= 1
    let revision = generation
    do {
      let reply = try await client.chat(.load(id))
      guard revision == generation, !stopped else { return }
      conversation = reply.conversation
      error = nil
      await transcript.show(conversation)
    } catch { if revision == generation, !stopped { self.error = error.localizedDescription } }
  }
  /// Returns whether the user's draft was accepted. The caller only clears it
  /// after Rust has persisted the user turn and delivered a stream receipt.
  public func send(_ text: String, runner: RunnerInfo) async -> Bool {
    guard !busy, !stopped else { return false }
    busy = true
    error = nil
    cancelRequested = false
    generation &+= 1
    let revision = generation
    do {
      let receipt = try await client.chat(
        .send(conversationId: conversation?.id, runner: runner, text: text))
      guard let id = receipt.streamId, let document = receipt.conversation else {
        throw ManagerError.core("The core did not return a chat receipt.")
      }
      if stopped {
        _ = try? await client.chat(.cancel(id))
        busy = false
        return true
      }
      streamID = id
      conversation = document
      if cancelRequested { _ = try? await client.chat(.cancel(id)) }
      await transcript.show(document)
      Task { await self.receive(id, revision: revision) }
      return true
    } catch {
      self.error = error.localizedDescription
      busy = false
      return false
    }
  }
  private func receive(_ id: String, revision: UInt64) async {
    // Only presentation deltas cross here, never provider credentials or raw
    // networking. Bound polling and await WebKit delivery before taking more.
    var textParts: [String: String] = [:]
    var reasoningParts: [String: String] = [:]
    do {
      while !stopped, generation == revision {
        let reply = try await client.chat(.poll(id))
        guard generation == revision, !stopped else { return }
        let events = reply.events ?? []
        for event in events {
          let key = String(format: "%010ld:%010ld", event.outputIndex, event.contentIndex)
          if event.kind == "reasoning" {
            reasoningParts[key, default: ""] += event.delta
          } else if event.kind == "text" {
            textParts[key, default: ""] += event.delta
          }
        }
        if !events.isEmpty, var document = conversation, let last = document.messages.indices.last {
          document.messages[last].content = [
            .init(
              type: "text",
              text: textParts.keys.sorted().compactMap { textParts[$0] }.joined(separator: "\n"))
          ]
          document.messages[last].reasoning = reasoningParts.keys.sorted().compactMap {
            reasoningParts[$0]
          }.joined(separator: "\n")
          document.messages[last].nativeTextParts = textParts
          document.messages[last].nativeReasoningParts = reasoningParts
          conversation = document
          await transcript.append(events, source: document)
        }
        if let done = reply.done {
          if let failure = done.saveError {
            throw ManagerError.core("The response could not be saved: \(failure)")
          }
          let saved = try await client.chat(.load(done.conversationId))
          guard generation == revision, !stopped else { return }
          conversation = saved.conversation
          await transcript.show(conversation)
          if done.status == "failed" { error = done.error ?? "The model failed to respond." }
          break
        }
        try await Task.sleep(for: .milliseconds(16))
      }
    } catch {
      _ = try? await client.chat(.cancel(id))
      if !stopped, generation == revision { self.error = error.localizedDescription }
    }
    if generation == revision {
      busy = false
      streamID = nil
      await refreshHistory()
    }
  }
  public func cancel() async {
    guard busy else { return }
    cancelRequested = true
    guard let streamID else { return }
    do { _ = try await client.chat(.cancel(streamID)) } catch {
      self.error = error.localizedDescription
    }
    // Continue draining to the durable cancelled terminal, including its tail.
  }
  public func shutdown() async {
    stopped = true
    generation &+= 1
    if let streamID { _ = try? await client.chat(.cancel(streamID)) }
    streamID = nil
    busy = false
    transcriptStorage?.close()
    transcriptStorage = nil
  }
}
