import AppKit
import Foundation
import UniformTypeIdentifiers

extension StudioWorkspace {
  @discardableResult public func prepareAudio(_ clip: StudioState.AudioClip) async -> Bool {
    await audioPlayback.prepare(clip) { [self] in
      try await audioMedia.playbackCopy(id: clip.id) { [self] in try await downloadAudio(clip) }
    }
  }
  public func audioWaveform(_ clip: StudioState.AudioClip) async -> AudioWaveform? {
    await audioMedia.waveform(id: clip.id) { [self] in try await downloadAudio(clip) }
  }
  public func seekAudio(_ clip: StudioState.AudioClip, at seconds: Double) async {
    guard !microphoneBusy, seconds.isFinite, await prepareAudio(clip),
      !Task.isCancelled, audioPlayback.clipId == clip.id
    else { return }
    audioPlayback.seek(seconds)
  }
  public func audioGuards(_ clip: StudioState.AudioClip) -> [StudioState.Speech.Guard] {
    var result: [StudioState.Speech.Guard] = []
    for message in state?.nativeTranscript?.messages ?? [] {
      guard let speech = message.speech, speech.clip?.id == clip.id else { continue }
      for notice in speech.guards where !result.contains(notice) { result.append(notice) }
    }
    return result
  }
  public func playAudio(_ clip: StudioState.AudioClip, at seconds: Double? = nil) async {
    guard !microphoneBusy, await prepareAudio(clip), audioPlayback.clipId == clip.id else { return }
    if let seconds {
      audioPlayback.seek(seconds)
      if !audioPlayback.playing { audioPlayback.toggle() }
    } else {
      audioPlayback.toggle()
    }
  }
  public func saveAudio(_ clip: StudioState.AudioClip) async {
    do {
      let url = try await downloadAudio(clip)
      defer { try? FileManager.default.removeItem(at: url) }
      let panel = NSSavePanel()
      panel.nameFieldStringValue = URL(fileURLWithPath: clip.name).lastPathComponent
      panel.title = "Save original recording"
      guard let window = presentationWindow ?? viewerWindow,
        await panel.beginSheetModal(for: window) == .OK, let destination = panel.url
      else { return }
      // The panel obtains explicit overwrite approval; Data.write(.atomic)
      // preserves the old destination until the copy is complete.
      try await Task.detached {
        try Data(contentsOf: url, options: .mappedIfSafe).write(to: destination, options: .atomic)
      }.value
    } catch { self.error = error.localizedDescription }
  }
  public func exportSpeech(_ target: StudioMessageTarget, format: String) async {
    do {
      var payload = target.payload(action: "export")
      payload["format"] = .string(format)
      let reply = try await command("transcriptExport", payload)
      guard let exported = reply["export"]?.object,
        let name = exported["name"]?.text, let text = exported["text"]?.text
      else { return }
      let panel = NSSavePanel()
      panel.nameFieldStringValue = name
      panel.title = "Export transcript"
      guard let window = presentationWindow ?? viewerWindow,
        await panel.beginSheetModal(for: window) == .OK, let destination = panel.url
      else { return }
      try await Task.detached { try Data(text.utf8).write(to: destination, options: .atomic) }.value
    } catch { self.error = error.localizedDescription }
  }
}
