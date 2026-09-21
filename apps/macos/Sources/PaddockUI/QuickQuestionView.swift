import AppKit
import SwiftUI

/// Input first. No duplicated app title, traffic lights or form-like header.
struct QuickQuestionView: View {
  @Bindable var draft: QuickQuestionModel
  @Bindable var workspace: WorkspaceModel
  let openStudio: () -> Void
  let openManager: () -> Void
  let close: () -> Void
  let resize: (CGFloat) -> Void
  let appearanceChanged: (WorkspaceAppearance) -> Void
  @AppStorage("workspaceAppearance") private var appearance: WorkspaceAppearance = .system
  @State private var window: NSWindow?
  @State private var editorHeight: CGFloat = 56
  @State private var targeted = false
  private var enabled: Bool { !draft.transferring && !workspace.quitting }
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      ZStack(alignment: .topLeading) {
        if draft.text.isEmpty {
          Text("Ask anything…").font(.system(size: StudioDraftEditor.textFontSize))
            .foregroundStyle(.secondary)
            .padding(.leading, StudioDraftEditor.horizontalTextPadding)
            .padding(.top, StudioDraftEditor.verticalTextPadding).allowsHitTesting(false)
        }
        StudioDraftEditor(
          text: $draft.text, onSend: send, onFiles: draft.addFiles,
          onImage: draft.addImage, onCancel: close, editorID: "quick-question-input",
          editorLabel: "Quick question", onWindow: { window = $0 },
          onHeight: { editorHeight = min(180, max(56, $0 - 16)) }
        )
        .frame(height: editorHeight)
      }
      if !draft.attachments.isEmpty {
        PaddockScrollView(.horizontal) {
          HStack(spacing: 6) {
            ForEach(draft.attachments) { file in
              HStack(spacing: 6) {
                Image(systemName: "paperclip")
                Text(file.name).lineLimit(1).frame(maxWidth: 140)
                Button("Remove \(file.name)", systemImage: "xmark") {
                  draft.attachments.removeAll { $0.id == file.id }
                }.labelStyle(.iconOnly).buttonStyle(.plain)
              }.font(.system(size: 11)).padding(7)
                .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 7))
            }
          }
        }.scrollIndicators(.hidden).frame(height: 30)
      }
      if let error = draft.error ?? workspace.chat.error {
        Text(error).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
          .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
          .accessibilityIdentifier("quick-question-error")
      }
      HStack(spacing: 10) {
        Button("Attach files", systemImage: "paperclip", action: chooseFiles)
          .labelStyle(.iconOnly).buttonStyle(QuickQuestionButtonStyle())
          .help("Attach files - review in Studio")
          .accessibilityIdentifier("quick-question-attach")
        modelPicker
        Spacer(minLength: 4)
        if draft.transferring || draft.loadingDrops { ProgressView().controlSize(.mini) }
        Button("esc", action: close).font(.system(size: 10)).foregroundStyle(.tertiary)
          .buttonStyle(.plain).help("Dismiss - your draft is kept")
          .accessibilityLabel("Dismiss quick question")
          .accessibilityIdentifier("quick-question-dismiss")
        Button(action: send) {
          Image(systemName: draft.attachments.isEmpty ? "arrow.up" : "arrow.up.right")
            .font(.system(size: 13, weight: .semibold)).frame(width: 30, height: 30)
            .foregroundStyle(canSend ? PaddockStyle.canvas : .secondary)
            .background(canSend ? PaddockStyle.accent : PaddockStyle.surface, in: Circle())
        }.buttonStyle(.plain).disabled(!canSend)
          .help(draft.attachments.isEmpty ? "Ask in Studio · Return" : "Review in Studio")
          .accessibilityLabel(draft.attachments.isEmpty ? "Ask in Studio" : "Review in Studio")
          .accessibilityIdentifier("quick-question-send")
      }
    }
    .padding(16).frame(maxWidth: .infinity, minHeight: 132).fixedSize(
      horizontal: false, vertical: true
    )
    .background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 18))
    .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(PaddockStyle.border))
    .clipShape(RoundedRectangle(cornerRadius: 18))
    // The NSPanel owns appearance, including live system inheritance. A
    // preferredColorScheme override can pin a hosting window after System.
    .tint(PaddockStyle.accent)
    .disabled(!enabled)
    .onGeometryChange(for: CGFloat.self) {
      $0.size.height
    } action: {
      resize($0)
    }
    .onChange(of: appearance, initial: true) { _, value in appearanceChanged(value) }
    .onExitCommand(perform: close)
    .onDrop(of: [.fileURL], isTargeted: $targeted, perform: draft.addDroppedFiles)
    .overlay {
      if targeted {
        RoundedRectangle(cornerRadius: 18).strokeBorder(
          PaddockStyle.border,
          style: StrokeStyle(lineWidth: 2, dash: [5])
        ).allowsHitTesting(false)
      }
    }
    .task {
      do {
        try await workspace.prepareDesktopChat()
        if draft.modelID.isEmpty {
          draft.modelID = QuickQuestionModel.suggestedModel(workspace.chat.state) ?? ""
        }
      } catch { draft.error = error.localizedDescription }
    }
  }
  private var canSend: Bool {
    enabled && draft.hasContent && !draft.loadingDrops && !draft.modelID.isEmpty
      && workspace.chat.ready
  }
  private var modelPicker: some View {
    Menu {
      ForEach(workspace.chat.state?.models.filter { $0.chat } ?? []) { model in
        Button {
          draft.modelID = model.id
        } label: {
          Label(
            "\(model.title) · \(model.provider)",
            systemImage: draft.modelID == model.id ? "checkmark" : "cube")
        }.disabled(model.status != "ok")
      }
      Divider()
      Button("Manage models…", action: openManager)
    } label: {
      HStack(spacing: 5) {
        Text(workspace.chat.state?.models.first { $0.id == draft.modelID }?.title ?? "Select model")
          .lineLimit(1).truncationMode(.middle)
        Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
      }.font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
    }.menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
      .accessibilityLabel("Quick question model").accessibilityIdentifier("quick-question-model")
  }
  private func send() {
    guard canSend else { return }
    Task { _ = await draft.handoff(to: workspace, openStudio: openStudio) }
  }
  private func chooseFiles() {
    let picker = NSOpenPanel()
    picker.canChooseDirectories = false
    picker.allowsMultipleSelection = true
    picker.prompt = "Attach"
    if let window {
      picker.beginSheetModal(for: window) { response in
        if response == .OK { draft.addFiles(picker.urls) }
      }
    }
  }
}

private struct QuickQuestionButtonStyle: ButtonStyle {
  func makeBody(configuration: Configuration) -> some View {
    configuration.label.font(.system(size: 14)).foregroundStyle(.secondary)
      .frame(width: 26, height: 28).contentShape(Rectangle())
      .background(
        configuration.isPressed ? PaddockStyle.surface : .clear,
        in: RoundedRectangle(cornerRadius: 6))
  }
}
