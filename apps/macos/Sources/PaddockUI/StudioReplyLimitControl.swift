import PaddockStudio
import SwiftUI

/// Used by both Conversation settings and the composer's context popover.
struct StudioReplyLimitControl: View {
  @Binding var draft: ReplyLimitDraft
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      Dropdown(
        title: "Reply limit", value: draft.automatic ? "Automatic" : "Custom", fillsWidth: true
      ) {
        Button("Automatic") { draft.automatic = true }
        Button("Custom") { draft.automatic = false }
      }.accessibilityIdentifier("reply-limit-mode")
      if !draft.automatic {
        HStack(spacing: 8) {
          TextField("Token limit", text: $draft.text)
            .textFieldStyle(.plain).monospacedDigit()
            .accessibilityLabel("Maximum reply tokens")
            .accessibilityIdentifier("reply-limit-tokens")
          Text("tokens").foregroundStyle(.secondary).accessibilityHidden(true)
        }.font(.system(size: 12)).padding(.horizontal, 9).frame(height: 30)
          .background(
            PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
          )
          .overlay(
            RoundedRectangle(cornerRadius: PaddockStyle.Radius.control).strokeBorder(
              PaddockStyle.border))
        if let validation = draft.validation {
          Text(validation).font(.caption).foregroundStyle(PaddockStyle.caution)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }.help(
      "Applies to text replies, including thinking. Automatic uses each model's available capacity. A custom value is an upper limit, not a target length."
    )
  }
}
