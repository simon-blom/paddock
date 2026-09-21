import PaddockClient
import PaddockTranscript
import SwiftUI
import WebKit

@testable import PaddockUI

struct LegacyStudioConversationView: View {
  @Bindable var chat: StudioChatModel
  @Binding var draft: StudioDraft
  let runners: [RunnerInfo]
  let canStart: Bool
  let onStart: () -> Void
  @Environment(\.colorScheme) private var colorScheme

  private var choices: [RunnerInfo] { runners.filter { $0.status == "ok" && $0.model != nil } }
  private var selected: RunnerInfo? { choices.first { $0.id == chat.selectedRunnerID } }

  var body: some View {
    VStack(spacing: 0) {
      if chat.conversation != nil {
        TranscriptView(session: chat.transcript)
          .overlay(alignment: .top) {
            if let error = chat.transcript.error {
              HStack {
                Text(error).font(.system(size: 12))
                Button("Reload renderer") { chat.transcript.reload() }.buttonStyle(
                  FlatButtonStyle())
              }.padding(12).background(PaddockStyle.surface)
            }
          }
      } else {
        Spacer(minLength: 32)
        Text("What would you like to work on?")
          .font(.system(size: 28, weight: .medium)).tracking(-0.6)
          .padding(.bottom, 26)
      }
      VStack(spacing: 12) {
        if let error = chat.error {
          Text(error).font(.system(size: 12)).foregroundStyle(.secondary)
            .textSelection(.enabled).accessibilityIdentifier("chat-error")
        }
        composer
        if chat.conversation == nil {
          Button("Start a model", systemImage: "plus", action: onStart)
            .buttonStyle(FlatButtonStyle()).disabled(!canStart)
        }
      }.frame(maxWidth: 760).padding(.horizontal, 28).padding(.bottom, 24)
      if chat.conversation == nil { Spacer(minLength: 32) }
    }.frame(maxWidth: .infinity, maxHeight: .infinity).background(PaddockStyle.canvas)
      .task {
        await chat.refreshHistory()
        chooseDefault()
      }
      .onChange(of: choices.map(\.id)) { _, _ in chooseDefault() }
      .task(id: colorScheme) {
        if chat.conversation != nil { await chat.transcript.setDark(colorScheme == .dark) }
      }
      .onChange(of: chat.conversation?.id) { _, value in
        if value != nil { Task { await chat.transcript.setDark(colorScheme == .dark) } }
      }
  }
  private var composer: some View {
    VStack(spacing: 12) {
      ZStack(alignment: .topLeading) {
        if draft.message.isEmpty {
          Text("Ask anything…").foregroundStyle(.secondary).padding(.leading, 5).padding(.top, 1)
            .allowsHitTesting(false)
        }
        TextEditor(text: $draft.message).scrollContentBackground(.hidden)
          .frame(height: chat.conversation == nil ? 98 : 68)
          .accessibilityLabel("Message draft").accessibilityIdentifier("studio-message")
      }.font(.system(size: 15))
      HStack(spacing: 12) {
        Button("Attach files", systemImage: "plus") {}.labelStyle(.iconOnly)
          .buttonStyle(QuietButtonStyle()).disabled(true).help("Attachments are not connected yet")
        Dropdown(
          title: "Model",
          value: selected.map { "\($0.title) · \($0.port)" } ?? "Select a running model"
        ) {
          ForEach(choices) { runner in
            Button("\(runner.title) · \(runner.port)") { chat.selectedRunnerID = runner.id }
          }
        }.disabled(chat.busy || choices.isEmpty)
        Spacer(minLength: 0)
        if chat.busy {
          Button("Stop", systemImage: "stop.fill") { Task { await chat.cancel() } }
            .buttonStyle(FlatButtonStyle()).accessibilityIdentifier("studio-stop")
        } else {
          Button("Send", systemImage: "arrow.up") { send() }
            .buttonStyle(FlatButtonStyle()).keyboardShortcut(.return, modifiers: .command)
            .disabled(
              selected == nil
                || draft.message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )
            .accessibilityIdentifier("studio-send")
        }
      }
    }.padding(16).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 16))
      .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(PaddockStyle.border))
  }
  private func chooseDefault() {
    // Auto-pick only before a user/previous selection exists. A stale pinned
    // port/PID never silently changes into another endpoint after a restart.
    if chat.selectedRunnerID == nil, choices.count == 1 { chat.selectedRunnerID = choices[0].id }
  }
  private func send() {
    guard let selected else { return }
    let text = draft.message
    Task {
      if await chat.send(text, runner: selected), draft.message == text { draft.message = "" }
    }
  }
}

private struct TranscriptView: NSViewRepresentable {
  let session: TranscriptSession
  func makeNSView(context: Context) -> WKWebView { session.webView }
  func updateNSView(_ view: WKWebView, context: Context) {}
}
