import PaddockStudio
import SwiftUI

/// The edit draft lives in the app-owned workspace, not in a lazy row. Leaving
/// the viewport or visiting Manager cannot destroy it or overwrite the composer.
struct NativeMessageEditor: View {
  @Bindable var chat: StudioWorkspace
  let edit: StudioMessageEdit
  @State private var height: CGFloat = 72
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Text("Edit message").font(.system(size: 12, weight: .medium))
      StudioDraftEditor(
        text: Binding(
          get: { chat.messageEdit?.text ?? edit.text }, set: { chat.messageEdit?.text = $0 }),
        onSend: submit, onCancel: chat.cancelMessageEdit, focusOnMount: true,
        editorID: "native-edit-text", editorLabel: "Edit message",
        onHeight: { height = $0 }
      ).frame(height: height).disabled(chat.busy)
      Text("Creates a new branch. The original conversation and attached files are kept.")
        .font(.system(size: 11)).foregroundStyle(.secondary)
      if let error = chat.error {
        Text(error).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled)
          .accessibilityIdentifier("native-edit-error")
      }
      HStack(spacing: 10) {
        Text("Return to send · ⇧Return for a new line · Esc to cancel")
          .font(.system(size: 10)).foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
        Spacer(minLength: 0)
        Button("Cancel", action: chat.cancelMessageEdit).buttonStyle(FlatButtonStyle())
          .disabled(chat.busy).accessibilityIdentifier("native-edit-cancel")
        Button(chat.messageMutation ? "Saving…" : "Send edit", action: submit)
          .modifier(PrimaryAction()).disabled(chat.busy || chat.messageEdit?.canSubmit != true)
          .accessibilityIdentifier("native-edit-send")
      }
    }.padding(16)
      .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 18))
      .overlay(RoundedRectangle(cornerRadius: 18).strokeBorder(PaddockStyle.border))
      .accessibilityIdentifier("native-message-editor")
  }
  private func submit() { Task { await chat.submitMessageEdit() } }
}
