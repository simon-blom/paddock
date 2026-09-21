import PaddockStudio
import SwiftUI

struct StudioPreferencesView: View {
  @Bindable var model: StudioPreferencesModel
  let busy: Bool
  var chat: StudioWorkspace?
  @State private var microphoneOptions = false
  @State private var confirmReload = false
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 22) {
        PageHeading(title: "Conversation") { EmptyView() }
        if let error = model.error {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
        }
        if let notice = model.notice { Text(notice).foregroundStyle(.secondary) }
        if model.loading { ProgressView().controlSize(.small) }
        if model.loaded, let layout = model.layout {
          VStack(alignment: .leading, spacing: 22) {
            ForEach(layout.sections) { section in
              setting(section, layout: layout)
                .accessibilityElement(children: .contain)
                .accessibilityIdentifier("studio-setting-\(section.id)")
              if section.id != layout.sections.last?.id { WorkspaceRule() }
            }
          }.disabled(model.saving || model.loading)
          if let validation = model.validation {
            Text(validation).foregroundStyle(PaddockStyle.caution)
          }
        }
        HStack {
          Button("Reload saved settings") {
            if model.dirty { confirmReload = true } else { Task { await model.load() } }
          }.disabled(model.saving || model.loading)
          Spacer()
          Button(model.saving ? "Saving…" : "Apply") { model.save() }
            .buttonStyle(FlatButtonStyle(primary: true)).disabled(
              !model.dirty || model.validation != nil || model.saving || model.loading || busy)
        }.buttonStyle(FlatButtonStyle())
        if busy {
          Text("Finish the response before applying preferences.").font(.caption).foregroundStyle(
            .secondary)
        }
      }.font(.system(size: 13)).padding(32).frame(maxWidth: 820).frame(maxWidth: .infinity)
    }.background(PaddockStyle.canvas).tint(PaddockStyle.accent).task { await model.load() }
      .confirmationDialog(
        "Discard your settings draft and reload?", isPresented: $confirmReload,
        titleVisibility: .visible
      ) {
        Button("Discard and reload", role: .destructive) {
          Task { await model.load(discard: true) }
        }
        Button("Keep editing", role: .cancel) {}
      }
  }
  @ViewBuilder private func setting(
    _ section: StudioSettingsLayout.Section, layout: StudioSettingsLayout
  ) -> some View {
    switch section.id {
    case "maxTokens":
      VStack(alignment: .leading, spacing: 10) {
        let index = layout.replyIndex(model.replyLimit)
        HStack {
          Text(section.title).fontWeight(.medium)
          Spacer()
          Text(layout.replyStops[index].label).monospacedDigit().foregroundStyle(.secondary)
        }
        StudioSamplingSlider(
          value: Binding(
            get: { Double(layout.replyIndex(model.replyLimit)) },
            set: { value in
              let index = min(layout.replyStops.count - 1, max(0, Int(value.rounded())))
              model.replyLimit = layout.replyStops[index].value.map(String.init) ?? ""
            }),
          bounds: 0...Double(max(1, layout.replyStops.count - 1)), step: 1,
          label: section.title
        ).frame(height: 18).disabled(layout.replyStops.count == 1)
          .help("Maximum tokens per reply, including thinking. Limited by the available context.")
        HStack {
          ForEach(layout.replyStops.indices, id: \.self) { i in
            Text(layout.replyStops[i].shortLabel).font(.system(size: 10)).monospacedDigit()
              .foregroundStyle(i == index ? .primary : .secondary)
            if i < layout.replyStops.count - 1 { Spacer(minLength: 0) }
          }
        }
      }
    case "maxToolCalls":
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          Text(section.title).fontWeight(.medium)
          Spacer()
          Dropdown(
            title: section.title,
            value: layout.toolStops.first { $0.value == (Int(model.toolLimit) ?? 0) }?.label
              ?? "\(model.toolLimit) tool calls"
          ) {
            ForEach(layout.toolStops) { stop in
              Button(stop.label) { model.toolLimit = stop.value == 0 ? "" : String(stop.value) }
            }
          }.frame(width: 180).help("Maximum tool calls per reply.")
        }
      }
    case "summarize":
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          Text(section.title).fontWeight(.medium)
          Spacer()
          Toggle(section.title, isOn: $model.summarize)
            .labelsHidden().toggleStyle(.switch).controlSize(.small)
            .help(
              "Summarize older messages when context fills. When off, the oldest messages are dropped."
            )
        }
      }
    case "microphone":
      VStack(alignment: .leading, spacing: 10) {
        Text(section.title).fontWeight(.medium)
        if let chat {
          StudioAudioInputSettings(chat: chat)
          Button("Audio settings…") { microphoneOptions = true }
            .buttonStyle(FlatButtonStyle())
            .popover(isPresented: $microphoneOptions) {
              StudioMicrophoneSettings(chat: chat).studioPopoverSurface()
            }
        }
      }
    case "mapTiles":
      VStack(alignment: .leading, spacing: 10) {
        HStack {
          Text(section.title).fontWeight(.medium)
          Spacer()
          Text(model.mapHost).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
        }
        explanation("Interactive maps share the photo's location with this tile server.")
        TextField("Follow the theme", text: $model.mapTiles)
          .textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel(section.title)
      }
    default:
      // New shared sections must not silently disappear before a native adapter lands.
      Text("\(section.title) is not yet available in the native app.").foregroundStyle(.secondary)
    }
  }
  private func explanation(_ text: String) -> some View {
    Text(text).font(.caption).foregroundStyle(.secondary).fixedSize(
      horizontal: false, vertical: true)
  }
}
