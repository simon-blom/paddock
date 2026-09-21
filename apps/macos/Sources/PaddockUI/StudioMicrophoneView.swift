import AppKit
import PaddockStudio
import SwiftUI

extension EnvironmentValues {
  @Entry var studioSpeechWorkspace: WorkspaceModel? = nil
}

struct StudioMicrophoneButton: View {
  @Bindable var chat: StudioWorkspace
  @Binding var draft: StudioDraft
  @Environment(\.studioSpeechWorkspace) private var workspace
  var compact = false
  @State private var settings = false
  private var audio: StudioState.Audio? { chat.state?.audio }
  private var title: String {
    audio?.menu?.needsSetup == true
      ? "Dictate" : StudioMicrophoneSettings.label(audio?.mode ?? "dictate")
  }
  var body: some View {
    HStack(spacing: 0) {
      Button {
        if chat.microphoneBusy || audio?.retryAvailable == true {
          stop()
        } else if audio?.menu?.needsSetup == true || audio?.jobs.isEmpty != false {
          settings = true
        } else {
          Task { await chat.startMicrophone() }
        }
      } label: {
        Image(systemName: chat.microphoneBusy ? "stop.fill" : "mic")
      }.buttonStyle(ComposerButtonStyle(active: chat.microphoneBusy))
        .disabled(
          !chat.ready || chat.uploading || chat.hasMessageEdit
            || (chat.busy && !chat.microphoneBusy) || workspace?.speech.busy == true
        )
        .help(
          chat.microphoneBusy
            ? "Stop microphone" : audio?.retryAvailable == true ? "Retry sending recording" : title
        )
        .accessibilityLabel(
          chat.microphoneBusy
            ? "Stop microphone" : audio?.retryAvailable == true ? "Retry sending recording" : title
        )
        .accessibilityIdentifier("composer-microphone")
      if chat.ready {
        Button {
          settings = true
        } label: {
          Image(systemName: "chevron.down").font(.system(size: 8))
        }.buttonStyle(.plain).frame(width: 18, height: 32).contentShape(Rectangle())
          .help("Audio settings")
          .accessibilityLabel("Audio settings").accessibilityIdentifier("microphone-options")
      }
    }
    .contextMenu {
      Button("Audio settings…") { settings = true }
      if chat.microphoneBusy || audio?.retryAvailable == true {
        Button("Discard recording", role: .destructive) { Task { await chat.cancelMicrophone() } }
      }
    }
    .popover(isPresented: $settings, arrowEdge: .top) {
      StudioMicrophoneSettings(chat: chat).studioPopoverSurface()
    }
    .task(id: audio?.shouldStop) { if audio?.shouldStop == true { stop() } }
  }
  private func stop() {
    let text = draft.message
    let transcriptionOnly = audio?.audioMode == true
    Task {
      if await chat.stopMicrophone(text: text), !transcriptionOnly, draft.message == text {
        draft.message = ""
      }
    }
  }
}

struct StudioMicrophoneSettings: View {
  @Bindable var chat: StudioWorkspace
  @Environment(\.studioSpeechWorkspace) private var workspace
  @Environment(\.dismiss) private var dismiss
  private var audio: StudioState.Audio? { chat.state?.audio }
  static func label(_ mode: String) -> String {
    switch mode {
    case "live": "Live"
    case "record": "Record and send"
    default: "Transcribe into the composer"
    }
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Text("Audio settings").fontWeight(.medium).padding(8)
      StudioAudioInputSettings(chat: chat)
      Divider()
      if let audio, let menu = audio.menu {
        if menu.needsSetup {
          Text(menu.setupMessage).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true).padding(8)
        } else {
          if menu.jobChoice {
            if menu.deviceChoice || menu.earChoice { heading("What the mic does") }
            if audio.audioMode {
              StudioPopoverChoice(
                title: "Live", subtitle: audio.liveBlocked ? audio.liveReason : nil,
                selected: audio.mode == "live"
              ) { change("mode", "live") }
              .disabled(!audio.jobs.contains("live") || chat.busy || chat.microphoneBusy)
            }
            if audio.jobs.contains("record") {
              StudioPopoverChoice(
                title: "Record and send",
                selected: audio.mode == "record"
              ) { change("mode", "record") }
              .disabled(chat.busy || chat.microphoneBusy)
            }
            if audio.jobs.contains("dictate") {
              StudioPopoverChoice(
                title: "Transcribe into the composer",
                selected: audio.mode == "dictate"
              ) { change("mode", "dictate") }
              .disabled(chat.busy || chat.microphoneBusy)
            }
          }
          if audio.mode == "dictate", !audio.transcribers.isEmpty {
            if menu.jobChoice { Divider() }
            heading("Heard by")
            ForEach(audio.transcribers) { model in
              StudioPopoverChoice(title: model.label, selected: audio.transcriber == model.id) {
                change("transcriber", model.id)
              }.disabled(chat.busy || chat.microphoneBusy)
            }
          }
        }
        if let rows = audio.speechModels, !rows.isEmpty {
          if menu.needsSetup || menu.jobChoice || menu.deviceChoice || menu.earChoice { Divider() }
          if !menu.needsSetup { heading("Speech models") }
          StudioSpeechModelRows(
            rows: rows,
            pendingPort: workspace?.speech.port
              ?? (workspace?.latestJob?.isActive == true ? workspace?.latestJob?.port : nil),
            blocked: workspace?.canSubmit != true || chat.busy || chat.microphoneBusy
          ) { row, start in
            workspace?.speech.act(row, start: start)
          }
          Divider()
        }
        if menu.needsSetup || audio.speechModels?.isEmpty == false {
          Button {
            dismiss()
            workspace?.request(.startSpeechModel)
          } label: {
            Label(menu.setupAction, systemImage: "plus").frame(
              maxWidth: .infinity, alignment: .leading
            )
            .padding(8).contentShape(Rectangle())
          }.buttonStyle(QuietButtonStyle()).disabled(
            workspace == nil || workspace?.canSubmit != true || chat.busy || chat.microphoneBusy
          )
          .accessibilityIdentifier("speech-setup")
        }
        if let error = audio.speechError, !error.isEmpty {
          Text(error).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        if !audio.error.isEmpty { Text(audio.error).foregroundStyle(.secondary) }
      } else {
        ProgressView().controlSize(.small).padding(8)
      }
      if let error = workspace?.speech.error {
        Text(error).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        if workspace?.speech.pending != nil {
          Button("Retry status") { workspace?.speech.retryStatus() }
            .disabled(workspace?.speech.checking == true)
        }
      }
      if let error = chat.error { Text(error).foregroundStyle(.secondary) }
    }.font(.system(size: 12)).padding(8).frame(width: 360)
      .task { await chat.perform("microphoneRefresh") }
  }
  private func heading(_ title: String) -> some View {
    Text(title).font(.system(size: 11)).foregroundStyle(.secondary).padding(.horizontal, 8)
  }
  private func change(_ key: String, _ value: String) {
    guard !chat.busy, !chat.microphoneBusy else { return }
    Task { await chat.perform("microphoneSettings", [key: .string(value)]) }
  }
}

/// Same input preference and explicit device-reveal action as web Settings.
/// No capture starts merely because this view or its popover is opened.
struct StudioAudioInputSettings: View {
  @Bindable var chat: StudioWorkspace
  private var audio: StudioState.Audio? { chat.state?.audio }
  private var selected: String {
    guard let audio, !audio.device.isEmpty else { return "System default" }
    return audio.devices.first { $0.id == audio.device }?.label ?? "Chosen microphone"
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      Dropdown(title: "Microphone", value: selected, fillsWidth: true) {
        Button("System default") { choose("") }
        ForEach(audio?.devices ?? []) { device in
          Button(device.label + (device.available == false ? " · Not connected" : "")) {
            choose(device.id)
          }
          .disabled(device.available == false)
        }
      }.disabled(chat.busy || chat.microphoneBusy || !chat.ready)
        .accessibilityIdentifier("audio-input-device")
      if audio?.devicesNamed != true {
        Button("Show my microphones") { Task { await chat.revealMicrophones() } }
          .buttonStyle(FlatButtonStyle()).disabled(chat.busy || chat.microphoneBusy || !chat.ready)
          .accessibilityIdentifier("audio-reveal-devices")
        Text("Allow microphone access to choose an input.")
          .font(.system(size: 11)).foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      if chat.microphoneBusy {
        Text("Stop the microphone to change its input or transcription settings.")
          .font(.system(size: 11)).foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      HStack {
        Button("Sound input…") { openSettings("com.apple.preference.sound?input") }
          .accessibilityIdentifier("audio-system-input")
        Spacer()
        Button("Microphone access…") {
          openSettings("com.apple.preference.security?Privacy_Microphone")
        }.accessibilityIdentifier("audio-system-permission")
      }.buttonStyle(QuietButtonStyle()).font(.system(size: 11))
      if let note = audio?.deviceNote, !note.isEmpty { Text(note).font(.caption) }
    }.padding(8)
  }
  private func choose(_ id: String) {
    Task { await chat.perform("microphoneSettings", ["device": .string(id)]) }
  }
  private func openSettings(_ pane: String) {
    if let url = URL(string: "x-apple.systempreferences:\(pane)") { NSWorkspace.shared.open(url) }
  }
}

struct StudioSpeechModelRows: View {
  let rows: [StudioState.Audio.SpeechModel]
  var pendingPort: UInt16?
  var blocked = false
  let act: (StudioState.Audio.SpeechModel, Bool) -> Void
  var body: some View {
    if rows.count > 4 {
      PaddockScrollView { content }.frame(height: 208)
    } else {
      content
    }
  }
  private var content: some View {
    VStack(spacing: 4) {
      ForEach(rows) { row in
        HStack(spacing: 8) {
          if let image = ProviderArtwork.image(for: row.vendor) {
            Image(nsImage: image).resizable().scaledToFit().frame(width: 14, height: 14)
          } else {
            Image(systemName: "mic").frame(width: 14)
          }
          VStack(alignment: .leading, spacing: 2) {
            Text(row.title).lineLimit(1).truncationMode(.tail).help(row.title)
            Text(pendingPort == row.port ? "working..." : row.status)
              .font(.system(size: 10)).foregroundStyle(.secondary)
          }.frame(maxWidth: .infinity, alignment: .leading)
          Button("Start") { act(row, true) }.disabled(blocked || !row.canStart)
            .accessibilityLabel("Start \(row.title)").accessibilityIdentifier(
              "speech-start-\(row.port)")
          Button("Stop") { act(row, false) }.disabled(blocked || !row.canStop)
            .accessibilityLabel("Stop \(row.title)").accessibilityIdentifier(
              "speech-stop-\(row.port)")
        }.padding(.horizontal, 8).frame(height: 48)
          .buttonStyle(FlatButtonStyle()).font(.system(size: 12))
      }
    }
  }
}

struct StudioMicrophoneStatus: View {
  let audio: StudioState.Audio
  let meter: StudioMicrophoneMeter
  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      if audio.busy {
        HStack(spacing: 8) {
          StudioMicrophoneLevelView(meter: meter, listening: audio.phase == "listening")
          Text(
            audio.phase == "finishing"
              ? "Finishing transcription…"
              : audio.arming
                ? "Starting microphone…"
                : audio.idle
                  ? "Listening · Nothing heard for a while"
                  : audio.mode == "record"
                    ? "Recording · \(Self.clock(audio.elapsed)) · \(Self.clock(audio.remaining)) left"
                    : "Listening…"
          )
          .monospacedDigit()
        }.accessibilityIdentifier("microphone-status")
      }
      if !audio.deviceNote.isEmpty { Text(audio.deviceNote) }
      if !audio.error.isEmpty { Text(audio.error).textSelection(.enabled) }
    }.font(.system(size: 11)).foregroundStyle(.secondary)
  }
  static func clock(_ seconds: Double) -> String {
    let n = Int(max(0, min(360_000, seconds.isFinite ? seconds : 0)))
    return "\(n / 60):\(String(format: "%02d", n % 60))"
  }
}
