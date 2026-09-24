import PaddockStudio
import SwiftUI

/// Mirrors the web thread's context boundary. This is relevant history state,
/// not a permanent help banner; raw messages remain visible and selectable.
struct NativeContextBoundary: View {
  let context: StudioState.NativeTranscript.Context
  @State private var expanded = false
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if !context.summary.isEmpty {
        DisclosureGroup(context.title, isExpanded: $expanded) {
          Text(context.summary).textSelection(.enabled).font(.system(size: 13))
            .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
        }
      } else if !context.title.isEmpty {
        Label(context.title, systemImage: "text.alignleft")
      }
      if context.working {
        HStack(spacing: 8) {
          ProgressView().controlSize(.mini)
          Text("Summarizing earlier messages…")
        }
      }
      if !context.error.isEmpty { Text(context.error).textSelection(.enabled) }
    }.font(.system(size: 12)).foregroundStyle(.secondary)
      .padding(12).frame(maxWidth: .infinity, alignment: .leading)
      .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 10))
      .accessibilityIdentifier("native-context-boundary")
  }
}
