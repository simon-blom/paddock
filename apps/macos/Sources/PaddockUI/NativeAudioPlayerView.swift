import PaddockStudio
import SwiftUI

/// Native SwiftUI controls backed by AVFoundation, not an HTML audio element.
struct NativeAudioPlayerView: View {
  let clip: StudioState.AudioClip
  @Bindable var workspace: StudioWorkspace
  @State private var waveform: AudioWaveform?
  @State private var waveformLoading = true
  @State private var seekTask: Task<Void, Never>?
  @State private var intendedPosition: Double?
  private var active: Bool { workspace.audioPlayback.clipId == clip.id }
  private var duration: Double {
    let value =
      active && workspace.audioPlayback.duration > 0
      ? workspace.audioPlayback.duration : waveform?.duration ?? clip.duration ?? 0
    return value.isFinite ? max(0, value) : 0
  }
  var body: some View {
    let player = workspace.audioPlayback
    VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 10) {
        Image(systemName: "waveform").foregroundStyle(.secondary)
        Text(verbatim: clip.name).font(.system(size: 12, weight: .medium)).lineLimit(1)
        Spacer(minLength: 4)
        if waveformLoading { ProgressView().controlSize(.mini).help("Reading waveform") }
        if let size = clip.size {
          Text(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file))
            .font(.system(size: 11)).foregroundStyle(.secondary)
        }
        Button("Save original recording", systemImage: "square.and.arrow.down") {
          Task { await workspace.saveAudio(clip) }
        }.labelStyle(.iconOnly).buttonStyle(.plain).help("Save original recording")
      }
      NativeAudioWaveformClock(
        clip: clip, workspace: workspace, waveform: waveform,
        duration: duration, intendedPosition: intendedPosition, onSeek: seek, onToggle: toggle)
      HStack(spacing: 12) {
        Button(
          active && player.playing ? "Pause recording" : "Play recording",
          systemImage: active && player.playing ? "pause.fill" : "play.fill"
        ) {
          Task { await workspace.playAudio(clip) }
        }.labelStyle(.iconOnly).buttonStyle(.plain).frame(width: 24, height: 26)
          .disabled(workspace.microphoneBusy || (active && player.loading))
          .accessibilityIdentifier("native-audio-play-\(clip.id)")
        if active && player.loading { ProgressView().controlSize(.small) }
        Button("Back five seconds", systemImage: "gobackward.5") {
          seek((intendedPosition ?? (active ? player.renderingPosition : 0)) - 5)
        }.labelStyle(.iconOnly).buttonStyle(.plain).disabled(
          duration <= 0 || workspace.microphoneBusy)
        Button("Forward five seconds", systemImage: "goforward.5") {
          seek((intendedPosition ?? (active ? player.renderingPosition : 0)) + 5)
        }.labelStyle(.iconOnly).buttonStyle(.plain).disabled(
          duration <= 0 || workspace.microphoneBusy)
        Text(
          "\(speechClock(intendedPosition ?? (active ? player.position : 0))) / \(duration > 0 ? speechClock(duration) : "--:--")"
        )
        .font(.system(size: 11)).monospacedDigit().foregroundStyle(.secondary).fixedSize()
        Spacer(minLength: 0)
        Menu {
          ForEach([0.75, 1, 1.25, 1.5, 2], id: \.self) { rate in
            Button(
              "\(rate.formatted())×", systemImage: player.rate == Float(rate) ? "checkmark" : ""
            ) {
              player.rate = Float(rate)
            }
          }
        } label: {
          Text("\(Double(player.rate).formatted())×").font(.system(size: 11))
        }
        .menuStyle(.borderlessButton).fixedSize().help("Playback speed")
        AudioVolumeControl(player: player)
      }
      if active, let error = player.error {
        Text(error).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled)
      } else if !waveformLoading && waveform == nil {
        Text("Waveform unavailable. You can still try playback.")
          .font(.system(size: 11)).foregroundStyle(.secondary)
      }
    }.padding(14).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 16))
      .accessibilityElement(children: .contain).accessibilityLabel("Audio player")
      .accessibilityIdentifier("native-audio-player-\(clip.id)")
      .task(id: clip.id) {
        seekTask?.cancel()
        intendedPosition = nil
        waveform = nil
        waveformLoading = true
        do { try await Task.sleep(for: .milliseconds(150)) } catch { return }
        let result = await workspace.audioWaveform(clip)
        guard !Task.isCancelled else { return }
        waveform = result
        waveformLoading = false
      }
      .onDisappear { seekTask?.cancel() }
  }
  private func toggle() { Task { await workspace.playAudio(clip) } }
  private func seek(_ seconds: Double) {
    guard duration > 0, !workspace.microphoneBusy else { return }
    let target = min(duration, max(0, seconds))
    if active && !workspace.audioPlayback.loading {
      seekTask?.cancel()
      intendedPosition = nil
      workspace.audioPlayback.seek(target)
      return
    }
    intendedPosition = target
    seekTask?.cancel()
    seekTask = Task {
      await workspace.seekAudio(clip, at: target)
      if !Task.isCancelled { intendedPosition = nil }
    }
  }
}

private struct NativeAudioWaveformClock: View {
  let clip: StudioState.AudioClip
  @Bindable var workspace: StudioWorkspace
  let waveform: AudioWaveform?
  let duration: Double
  let intendedPosition: Double?
  var onSeek: (Double) -> Void
  var onToggle: () -> Void
  var body: some View {
    let player = workspace.audioPlayback
    let active = player.clipId == clip.id
    let notices = workspace.audioGuards(clip)
    let playing = active && player.playing
    let pausedPosition = active && !playing ? player.position : 0
    TimelineView(.animation(minimumInterval: 1.0 / 60, paused: !playing)) { _ in
      NativeWaveformTrack(
        waveform: waveform, duration: duration,
        position: intendedPosition ?? (playing ? player.renderingPosition : pausedPosition),
        notices: notices, enabled: !workspace.microphoneBusy,
        onSeek: onSeek, onToggle: onToggle)
    }.frame(height: 48)
  }
}

private struct AudioVolumeControl: View {
  @Bindable var player: StudioAudioPlayback
  @State private var expanded = false
  var body: some View {
    Button(
      player.muted || player.volume == 0 ? "Unmute recording" : "Mute recording",
      systemImage: player.muted || player.volume == 0 ? "speaker.slash" : "speaker.wave.2"
    ) {
      if player.volume == 0 {
        player.volume = 1
        player.muted = false
      } else {
        player.muted.toggle()
      }
    }.labelStyle(.iconOnly).buttonStyle(.plain)
    Button("Volume", systemImage: "chevron.down") { expanded.toggle() }
      .labelStyle(.iconOnly).font(.system(size: 9)).buttonStyle(.plain).help("Volume")
      .popover(isPresented: $expanded, arrowEdge: .top) {
        VStack(alignment: .leading, spacing: 10) {
          Text("Volume").font(.system(size: 12, weight: .medium))
          Slider(
            value: Binding(
              get: { player.muted ? 0 : Double(player.volume) },
              set: {
                player.volume = Float($0)
                if $0 > 0 { player.muted = false }
              }), in: 0...1
          ).tint(.primary).accessibilityLabel("Playback volume")
        }.padding(14).frame(width: 200).studioPopoverSurface()
      }
  }
}

func speechClock(_ seconds: Double) -> String {
  guard seconds.isFinite else { return "0:00" }
  let value = Int(max(0, min(360_000, seconds)))
  return String(format: "%d:%02d", value / 60, value % 60)
}
