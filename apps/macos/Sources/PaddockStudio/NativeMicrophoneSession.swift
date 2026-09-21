import Foundation
import PaddockClient
import PaddockConversationCore

@MainActor final class NativeMicrophoneSession {
  typealias V = ConversationValue
  typealias O = [String: V]
  let runtime: NativeStudioRuntime
  let transport: NativeConversationTransport
  private let capture: NativeAudioCapture
  private let meter: StudioMicrophoneMeter
  private var pump: Task<Void, Never>?
  private var readers: [Task<Void, Never>] = []
  private var sockets: [String: NativeRealtimeConnection] = [:]
  private var mapping: [String: String] = [:]
  private var configured = Set<String>(), drainable = Set<String>()
  private var progress: [String: NativeSpeechProgress] = [:]
  private var laneFailures: [String: String] = [:]
  private var transcripts: [String: String] = [:], partial: [String: String] = [:]
  private var transcriptMetadata: [String: O] = [:]
  private var state: O = [:]
  private var mode = "dictate", stopping = false, cancelled = false
  private var failure: String?
  private var clip: O?
  private var stoppedFile: URL?
  private var lastMeter = ContinuousClock.now
  private var nextDictation = 0
  private var utteranceSamples = 0
  private var utteranceLimit = NativeSpeechPolicy.maximumUtteranceSeconds
  init(
    runtime: NativeStudioRuntime, transport: NativeConversationTransport,
    meter: StudioMicrophoneMeter
  ) {
    self.runtime = runtime
    self.transport = transport
    self.meter = meter
    capture = NativeAudioCapture(meter: meter)
  }
  func start() async throws {
    state = await runtime.audioPresentation()
    mode = state["mode"]?.string ?? "dictate"
    state["phase"] = .string("starting")
    state["session"] = .string(UUID().uuidString)
    state["dictation"] = .array([])
    state["provisional"] = .string("")
    state["error"] = .string("")
    await runtime.setAudio(state)
    do {
      if mode != "record" {
        let targets = try await runtime.audioTargets(dictation: mode == "dictate")
        utteranceLimit =
          targets.compactMap { $0["utteranceLimit"]?.double }.min()
          ?? NativeSpeechPolicy.maximumUtteranceSeconds
        if mode == "live" { mapping = try await runtime.beginLive() }
        for target in targets {
          let id = target["id"]!.string!
          let port = UInt16(target["port"]!.integer!)
          let socket = try await transport.realtime(port: port)
          sockets[id] = socket
          progress[id] = NativeSpeechProgress()
          if target["drain"]?.bool == true { drainable.insert(id) }
          let transcription = target["transcription"]?.object ?? [:]
          try await socket.send([
            "type": .string("session.update"),
            "session": .object([
              "type": .string("transcription"),
              "audio": .object([
                "input": .object([
                  "format": .object(["type": .string("audio/pcm"), "rate": .number(16000)]),
                  "turn_detection": .object([
                    "type": .string("server_vad"), "silence_duration_ms": .number(700),
                    "prefix_padding_ms": .number(600),
                    "idle_timeout_ms": .number(mode == "dictate" ? 5000 : 20000),
                  ]), "transcription": .object(transcription),
                ])
              ]),
            ]),
          ])
          readers.append(
            Task { [weak self] in
              do {
                while !Task.isCancelled {
                  let event = try await socket.receive()
                  await self?.receive(event, modelID: id)
                }
              } catch {
                guard let self, !self.cancelled, !Task.isCancelled else { return }
                await self.failLane(id, error.localizedDescription)
              }
            })
        }
        let readyDeadline = ContinuousClock.now + .seconds(15)
        while !Set(sockets.keys).isSubset(of: configured), failure == nil,
          ContinuousClock.now < readyDeadline
        {
          try await Task.sleep(for: .milliseconds(20))
        }
        guard Set(sockets.keys).isSubset(of: configured) else {
          throw ManagerError.core(
            failure ?? "A speech model did not accept the live session settings")
        }
      }
      if let failure { throw ManagerError.core(failure) }
      let frames = try await capture.start(
        deviceID: state["device"]?.string ?? "",
        maximumSeconds: state["limit"]?.double ?? NativeSpeechPolicy.recordingSeconds)
      guard !cancelled else {
        _ = try? await capture.stop()
        throw CancellationError()
      }
      if let failure { throw ManagerError.core(failure) }
      state["phase"] = .string("listening")
      await runtime.setAudio(state)
      pump = Task { [weak self] in
        guard let self else { return }
        do {
          for try await frame in frames {
            try Task.checkCancellation()
            // Await every lane before taking another bounded frame. No lane
            // gets a different conversion, dropping policy or sample sequence.
            let payload: O = [
              "type": .string("input_audio_buffer.append"),
              "audio": .string(frame.pcm.base64EncodedString()),
            ]
            await sendToLanes(payload)
            if progress.values.contains(where: { $0.speaking != nil }) {
              utteranceSamples += frame.pcm.count / 2
              if Double(utteranceSamples) >= utteranceLimit * Double(NativeSpeechPolicy.sampleRate)
              {
                utteranceSamples = 0
                await sendToLanes(["type": .string("input_audio_buffer.commit")])
              }
            } else {
              utteranceSamples = 0
            }
            let limit = state["limit"]?.double ?? NativeSpeechPolicy.recordingSeconds
            if ContinuousClock.now - lastMeter >= .milliseconds(100) {
              // A low-rate diagnostic projection only. The native meter reads
              // capture directly, independent of network backpressure/publish.
              state["levels"] = .array(meter.levels().map { .number(Decimal($0)) })
              state["elapsed"] = .number(Decimal(frame.elapsed))
              state["remaining"] = .number(Decimal(max(0, limit - frame.elapsed)))
              lastMeter = .now
              await runtime.setAudio(state)
            }
            if frame.elapsed >= limit {
              Task { _ = try? await self.stop(text: await runtime.audioDraftText()) }
              break
            }
          }
          if !cancelled, !stopping {
            Task { _ = try? await self.stop(text: await runtime.audioDraftText()) }
          }
        } catch { if !cancelled && !stopping { fail(error.localizedDescription) } }
      }
    } catch {
      failure = error.localizedDescription
      await cancel()
      throw error
    }
  }
  private func receive(_ event: O, modelID: String) async {
    guard !cancelled, laneFailures[modelID] == nil else { return }
    guard progress[modelID, default: NativeSpeechProgress()].receive(event) else { return }
    switch event["type"]?.string {
    case "session.updated": configured.insert(modelID)
    case "conversation.item.input_audio_transcription.delta":
      let delta = event["delta"]?.string ?? ""
      guard (partial[modelID]?.utf8.count ?? 0) + delta.utf8.count <= 256 * 1024 else {
        fail("The speech response exceeded the utterance limit. The recording is retained.")
        return
      }
      partial[modelID, default: ""] += delta
    case "conversation.item.input_audio_transcription.completed":
      let text = event["transcript"]?.string ?? event["text"]?.string ?? partial[modelID] ?? ""
      transcripts[modelID, default: ""] +=
        (transcripts[modelID]?.isEmpty == false ? " " : "") + text
      partial[modelID] = ""
      transcriptMetadata[modelID] = NativeSpeechMetadata.appendingLive(
        event, to: transcriptMetadata[modelID] ?? [:])
      if mode == "dictate", !text.isEmpty {
        var items = state["dictation"]?.array ?? []
        guard
          items.reduce(0, { $0 + ($1["text"]?.string?.utf8.count ?? 0) }) + text.utf8.count <= 48
            * 1024
        else {
          fail("Dictation could not be delivered to the composer. Stop and try again.")
          return
        }
        let next = nextDictation
        nextDictation += 1
        items.append(.object(["index": .number(Decimal(next)), "text": .string(text)]))
        state["dictation"] = .array(items)
      }
    case "error":
      await failLane(modelID, event["error"]?["message"]?.string ?? "Speech model failed")
    case "input_audio_buffer.timeout_triggered":
      if mode == "dictate", !transcripts.values.allSatisfy(\.isEmpty), !stopping {
        Task { _ = try? await self.stop(text: "") }
      }
    default: break
    }
    if mode == "dictate" {
      state["provisional"] = .string(failure == nil ? partial[modelID] ?? "" : "")
    } else if let mid = mapping[modelID] {
      let complete = transcripts[modelID] ?? ""
      let pending = partial[modelID] ?? ""
      try? await runtime.liveUpdate(
        messageID: mid,
        text: complete + (!complete.isEmpty && !pending.isEmpty ? " " : "") + pending,
        transcript: transcriptMetadata[modelID] ?? [:])
    }
    await runtime.setAudio(state)
  }
  private func sendToLanes(_ event: O) async {
    let lanes = sockets
    await withTaskGroup(of: (String, String?).self) { group in
      for (id, socket) in lanes {
        group.addTask {
          do {
            try await socket.send(event)
            return (id, nil)
          } catch { return (id, error.localizedDescription) }
        }
      }
      for await (id, error) in group { if let error { await failLane(id, error) } }
    }
  }
  private func failLane(_ id: String, _ message: String) async {
    guard laneFailures[id] == nil else { return }
    laneFailures[id] = message
    progress[id]?.fail()
    drainable.remove(id)
    if let socket = sockets.removeValue(forKey: id) { await socket.close() }
    if let mid = mapping[id] { try? await runtime.liveFailure(messageID: mid, error: message) }
    if mode != "live" || sockets.isEmpty { fail(message) }
  }
  private func fail(_ message: String) {
    guard failure == nil else { return }
    failure = message
    state["error"] = .string(message)
    state["provisional"] = .string("")
    Task { _ = try? await stop(text: "") }
  }
  func stop(text: String) async throws -> Bool {
    guard !stopping, !cancelled else { return false }
    stopping = true
    defer { stopping = false }
    state["phase"] = .string("finishing")
    await runtime.setAudio(state)
    do {
      let file: URL
      if let stoppedFile {
        file = stoppedFile
      } else {
        file = try await capture.stop()
        stoppedFile = file
      }
      await pump?.value
      pump = nil
      for id in drainable { progress[id]?.beginDrain() }
      for (id, socket) in sockets {
        do {
          if drainable.contains(id) {
            try await socket.send(["type": .string("input_audio_buffer.drain")])
          } else if progress[id]?.speaking != nil {
            try await socket.send(["type": .string("input_audio_buffer.commit")])
          }
        } catch { await failLane(id, error.localizedDescription) }
      }
      let deadline = ContinuousClock.now + .seconds(NativeSpeechPolicy.finalizationSeconds)
      while progress.values.contains(where: \.waiting), ContinuousClock.now < deadline,
        failure == nil
      {
        try await Task.sleep(for: .milliseconds(25))
      }
      for id in progress.keys.filter({ progress[$0]?.waiting == true }) {
        await failLane(
          id, "This speech model did not finish the final utterance. The recording is retained.")
      }
      await releaseSockets()
      state["provisional"] = .string("")
      if mode != "dictate" {
        if clip == nil {
          let data = try await Task.detached { try Data(contentsOf: file, options: .mappedIfSafe) }
            .value
          let id = UUID().uuidString
          _ = try await transport.bytes(
            "api/attachments/\(id)", method: "PUT", body: data, contentType: "audio/wav",
            query: ["name": "Recording.wav"])
          clip = [
            "type": .string("audio"), "attachmentId": .string(id), "name": .string("Recording.wav"),
            "mime": .string("audio/wav"), "size": .number(Decimal(data.count)),
            "durationS": .number(Decimal(max(0, data.count - 44)) / 32000),
            "language": state["language"] ?? .string("auto"),
          ]
        }
        if mode == "live" {
          try await runtime.finishLive(
            messages: Array(mapping.values), clip: clip, failure: failure)
        } else if let clip {
          let metadata: O = [
            "id": clip["attachmentId"]!, "name": clip["name"]!, "mime": clip["mime"]!,
            "size": clip["size"]!, "durationS": clip["durationS"]!,
          ]
          state["phase"] = .string("idle")
          await runtime.setAudio(state)
          _ = try await runtime.command("stage", metadata)
          let reply = try await runtime.command(
            "send",
            [
              "text": .string(text),
              "attachments": .array([.object(["id": clip["attachmentId"]!])]),
            ])
          if reply["accepted"]?.bool == true {
            state["phase"] = .string("idle")
            state["retryAvailable"] = .bool(false)
            state["attachment"] = nil
            await runtime.setAudio(state)
            removeTemporary()
            return true
          }
        }
      }
      state["phase"] = .string("idle")
      state["levels"] = .array([])
      state["error"] = .string(failure ?? "")
      await runtime.setAudio(state)
      removeTemporary()
      return mode == "live"
    } catch {
      await releaseSockets()
      state["provisional"] = .string("")
      state["phase"] = .string("idle")
      state["error"] = .string(error.localizedDescription)
      state["retryAvailable"] = .bool(true)
      if let clip { state["attachment"] = .object(clip) }
      await runtime.setAudio(state)
      // Retain the task-owned WAV and any uploaded original for retry.
      throw error
    }
  }
  func acknowledge(session: String, index: Int) async {
    guard state["session"]?.string == session else { return }
    // Retain monotonically increasing IDs when the visible queue is drained.
    state["dictation"] = .array(
      (state["dictation"]?.array ?? []).filter { ($0["index"]?.integer ?? -1) > index })
    await runtime.setAudio(state)
  }
  func cancel() async {
    cancelled = true
    state["provisional"] = .string("")
    _ = try? await capture.stop()
    pump?.cancel()
    await pump?.value
    pump = nil
    await releaseSockets()
    if !mapping.isEmpty {
      try? await runtime.finishLive(
        messages: Array(mapping.values), clip: clip, failure: failure ?? "Recording cancelled")
    }
    state["phase"] = .string("idle")
    state["levels"] = .array([])
    await runtime.setAudio(state)
    removeTemporary()
  }
  private func releaseSockets() async {
    for reader in readers { reader.cancel() }
    readers = []
    for socket in sockets.values { await socket.close() }
    sockets = [:]
  }
  private func removeTemporary() {
    try? FileManager.default.removeItem(at: capture.url)
    stoppedFile = nil
  }
}
