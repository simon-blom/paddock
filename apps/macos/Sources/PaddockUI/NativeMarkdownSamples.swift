import PaddockNativeMarkdown
import SwiftUI

/// Deterministic fixtures never enter a conversation or invoke a model.
struct NativeMarkdownSamples: View {
  @Environment(\.dismiss) private var dismiss
  @State private var fixture = "showcase"
  @State private var text = NativeMarkdownFixtures.showcase
  @State private var visible = NativeMarkdownFixtures.showcase
  @State private var streaming = false
  @State private var sourceVisible = false
  @State private var replay: Task<Void, Never>?
  @State private var renderID = UUID()
  var body: some View {
    VStack(spacing: 0) {
      sampleToolbar
      Divider()
      HStack(spacing: 0) {
        if sourceVisible {
          PaddockTextEditor(text: $text).font(.system(size: 12, design: .monospaced))
            .scrollContentBackground(.hidden).padding(12).frame(width: 320)
          Divider()
        }
        sampleContent
      }
    }.frame(width: 1000, height: 740).background(PaddockStyle.canvas)
      .onChange(of: fixture) { _, value in
        selectFixture(value)
      }
      .onChange(of: text) { _, value in
        replay?.cancel()
        streaming = false
        visible = value
      }
      .onDisappear { replay?.cancel() }
  }
  private func selectFixture(_ value: String) {
    switch value {
    case "diagrams": text = NativeMarkdownFixtures.diagrams
    case "long": text = NativeMarkdownFixtures.long
    case "message": text = NativeMessageSample.text
    default: text = NativeMarkdownFixtures.showcase
    }
  }

  private var sampleToolbar: some View {
    HStack(spacing: 12) {
      Text("Native rendering samples").font(.headline)
      Spacer()
      Picker("Fixture", selection: $fixture) {
        Text("Message layout").tag("message")
        Text("Markdown").tag("showcase")
        Text("Diagrams").tag("diagrams")
        Text("Long response").tag("long")
      }.labelsHidden().frame(width: 170)
      Button(sourceVisible ? "Hide source" : "Edit source") { sourceVisible.toggle() }
      Button(streaming ? "Stop" : "Replay stream") { play() }
      Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
    }.buttonStyle(FlatButtonStyle()).padding(18)
  }

  private var sampleContent: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 20) {
        if fixture == "message",
          let sample = try? NativeMessageSample.message(text: visible, streaming: streaming)
        {
          Text("Sample layout · illustrative statistics, not a benchmark")
            .font(.system(size: 11)).foregroundStyle(.secondary)
          NativeStudioMessage(message: sample).equatable().id(renderID)
        } else {
          NativeMarkdown(visible, streaming: streaming).equatable().id(renderID)
        }
      }.font(.system(size: 15)).frame(maxWidth: 760, alignment: .leading)
        .padding(28).frame(maxWidth: .infinity, alignment: .center)
    }
  }

  private func play() {
    replay?.cancel()
    if streaming {
      streaming = false
      return
    }
    let characters = Array(text)
    renderID = UUID()
    visible = ""
    streaming = true
    replay = Task { @MainActor in
      for end in stride(from: 24, to: characters.count + 24, by: 24) {
        do { try await Task.sleep(for: .milliseconds(35)) } catch { return }
        visible = String(characters.prefix(end))
      }
      streaming = false
    }
  }
}
