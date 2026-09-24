import AppKit
import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI

struct NativeDocumentResult: View {
  let result: StudioState.NativeTranscript.Message.DocumentResult
  var selectionPrefix = "document"
  var onOpenDocument: ((String, String) -> Void)? = nil
  var workspace: StudioWorkspace? = nil
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      if !result.facts.isEmpty {
        Text(result.facts.map { "\($0.label): \($0.value)" }.joined(separator: " · "))
          .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
      }
      ForEach(result.pages) { page in
        VStack(alignment: .leading, spacing: 10) {
          HStack {
            Text("Page \(page.number ?? page.id)").fontWeight(.medium)
            if page.state == "reading" { ProgressView().controlSize(.mini) }
            Text(stateLabel(page.state)).foregroundStyle(.secondary)
            Spacer()
            if let source = page.sourceID, let attachment = page.attachmentID, let onOpenDocument {
              Button("Open original", systemImage: "doc.text.magnifyingglass") {
                onOpenDocument(source, attachment)
              }
              .labelStyle(.iconOnly).buttonStyle(.plain).help(page.name ?? "Open original")
            }
            Button("Copy page", systemImage: "doc.on.doc") {
              NSPasteboard.general.clearContents()
              NSPasteboard.general.setString(page.text, forType: .string)
            }.labelStyle(.iconOnly).buttonStyle(.plain)
          }.font(.system(size: 12))
          if !page.note.isEmpty {
            if page.state == "error", page.note.trimmingCharacters(in: .whitespaces).hasPrefix("{")
            {
              NativeResponseErrorView(error: page.note)
            } else {
              Text(page.note).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
            }
          }
          NativeMarkdown(
            page.text, streaming: page.state == "reading", unsureWords: page.unsure.map(\.label)
          )
          .environment(\.conversationTextID, selectionPrefix + "/page-\(page.id)")
          if !page.unsure.isEmpty {
            Text(
              "Low confidence: "
                + page.unsure.map { "\($0.label) (\($0.value))" }.joined(separator: ", ")
            )
            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
          }
          if !page.regions.isEmpty {
            if let workspace, let source = page.sourceID, let attachment = page.attachmentID,
              let onOpenDocument
            {
              NativeDocumentFigures(
                page: page,
                load: {
                  try await workspace.documentMedia.image(
                    .init(attachmentID: attachment, pdfPage: page.pdfPage)
                  ) {
                    try await workspace.downloadOriginal(attachment)
                  }
                }, onOpen: { onOpenDocument(source, attachment) })
            } else {
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
      }
    }.accessibilityIdentifier("native-document-result")
  }
  private func stateLabel(_ state: String) -> String {
    switch state {
    case "reading": "Reading…"
    case "queued": "Queued"
    case "done": "Done"
    case "review": "Needs review"
    case "error": "Failed"
    default: state.capitalized
    }
  }
}
