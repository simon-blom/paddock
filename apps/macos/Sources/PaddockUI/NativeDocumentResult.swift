import AppKit
import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI

struct NativeDocumentResult: View {
  let result: StudioState.NativeTranscript.Message.DocumentResult
  var selectionPrefix = "document"
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text(result.facts.map { "\($0.label): \($0.value)" }.joined(separator: " · "))
        .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
      ForEach(result.pages) { page in
        VStack(alignment: .leading, spacing: 10) {
          HStack {
            Text("Page \(page.id)").fontWeight(.medium)
            Text(page.state).foregroundStyle(.secondary)
            Spacer()
            Button("Copy page", systemImage: "doc.on.doc") {
              NSPasteboard.general.clearContents()
              NSPasteboard.general.setString(page.text, forType: .string)
            }.labelStyle(.iconOnly).buttonStyle(.plain)
          }.font(.system(size: 12))
          if !page.note.isEmpty {
            Text(page.note).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
          }
          NativeMarkdown(page.text, streaming: page.state == "reading")
            .environment(\.conversationTextID, selectionPrefix + "/page-\(page.id)")
          if !page.unsure.isEmpty {
            Text(
              "Low confidence: "
                + page.unsure.map { "\($0.label) (\($0.value))" }.joined(separator: ", ")
            )
            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          }
          if !page.regions.isEmpty {
            DisclosureGroup("\(page.regions.count) grounded regions") {
              ForEach(Array(page.regions.enumerated()), id: \.offset) { _, region in
                VStack(alignment: .leading) {
                  Text(region.label).fontWeight(.medium)
                  if !region.text.isEmpty { Text(region.text) }
                  Text(
                    "Bounds: "
                      + region.boxes.map {
                        $0.map { String(format: "%g", $0) }.joined(separator: ", ")
                      }.joined(separator: "; ")
                  )
                  .foregroundStyle(.secondary)
                }.font(.caption).textSelection(.enabled).padding(.vertical, 4)
              }
            }
          }
        }
      }
    }.accessibilityIdentifier("native-document-result")
  }
}
