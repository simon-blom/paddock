import AVFoundation
import Foundation
import PaddockClient
import PaddockConversationCore

extension StudioWorkspace {
  func audioCommand(_ kind: String, _ payload: [String: ConversationValue]) async throws -> [String:
    StudioValue]
  {
    guard let runtime, let nativeTransport else {
      throw ManagerError.core("Native audio is not ready")
    }
    switch kind {
    case "microphoneStart":
      let session = NativeMicrophoneSession(
        runtime: runtime, transport: nativeTransport, meter: microphoneMeter)
      microphone = session
      try await session.start()
    case "microphoneStop":
      let accepted = try await microphone?.stop(text: payload["text"]?.string ?? "") ?? false
      return ["accepted": .bool(accepted)]
    case "microphoneCancel":
      await microphone?.cancel()
      microphone = nil
    case "microphoneSettings": try await runtime.configureAudio(payload)
    case "microphoneRefresh", "microphoneDevices":
      var state = await runtime.audioPresentation()
      if AVCaptureDevice.authorizationStatus(for: .audio) == .authorized {
        let devices = NativeAudioCapture.devices()
        state["devices"] = .array(
          devices.map {
            .object(
              $0.mapValues(ConversationValue.string).merging(["available": .bool(true)]) { _, new in
                new
              })
          })
        state["devicesNamed"] = .bool(true)
      }
      await runtime.setAudio(state)
    case "dictationAck":
      await microphone?.acknowledge(
        session: payload["session"]?.string ?? "", index: payload["index"]?.integer ?? -1)
    default: throw ManagerError.core("Unknown native audio action")
    }
    return [:]
  }
  public var pendingDictation: [StudioState.Audio.Item] {
    state?.audio?.dictation.filter { $0.index > dictatedThrough } ?? []
  }

  /// Exactly one native permission request, only in response to the mic button.
  /// AVFoundation is the only capture path; viewers are always denied media.
  public func startMicrophone() async {
    await requestMicrophone("microphoneStart")
  }

  /// Same explicit, trusted permission gate as Dictate. Enumeration briefly
  /// opens and immediately releases the input; it never records or transcribes.
  public func revealMicrophones() async {
    await requestMicrophone("microphoneDevices")
  }

  private func requestMicrophone(_ command: String) async {
    guard ready, !busy, !uploading, !hasMessageEdit else { return }
    audioPlayback.reset()
    microphoneEpoch += 1
    let ticket = microphoneEpoch
    microphoneStarting = true
    defer {
      if ticket == microphoneEpoch {
        microphoneStarting = false
        captureRequested = false
      }
    }
    let allowed: Bool
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized: allowed = true
    case .notDetermined: allowed = await AVCaptureDevice.requestAccess(for: .audio)
    default: allowed = false
    }
    guard ticket == microphoneEpoch else { return }
    guard allowed else {
      error =
        "Microphone access is off. Enable Paddock in System Settings > Privacy & Security > Microphone."
      return
    }
    captureRequested = true
    await perform(command)
  }

  /// Record submits through the same durable path as an attached audio file.
  /// Dictation never sends a message; only finalized utterances enter the draft.
  public func stopMicrophone(text: String) async -> Bool {
    if microphoneStarting {
      await cancelMicrophone()
      return false
    }
    do {
      let result = try await command("microphoneStop", ["text": .string(text)])
      let accepted = result["accepted"]?.boolean == true
      if accepted { attachments.removeAll() }
      return accepted
    } catch {
      self.error = error.localizedDescription
      return false
    }
  }

  public func cancelMicrophone() async {
    if let clip = state?.audio?.attachment { attachments.removeAll { $0.id == clip.attachmentId } }
    microphoneEpoch += 1
    microphoneStarting = false
    captureRequested = false
    await perform("microphoneCancel")
  }

  public func acknowledgeDictation(session: String, index: Int) {
    guard session == dictationSession, index > dictatedThrough else { return }
    // Mark locally before the bridge round trip: remount/retry cannot insert twice.
    dictatedThrough = index
    Task {
      await perform("dictationAck", ["session": .string(session), "index": .number(Double(index))])
    }
  }
}
