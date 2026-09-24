import PaddockStudio
import SwiftUI

/// Native controls, shared upload and send path. Editing never reads or
/// re-rasterizes the PDF; only its per-attachment metadata changes.
struct StudioAttachmentOptions: View {
  @Binding var attachment: StudioAttachment
  let canRasterPDF: Bool
  var capabilities: StudioState.Capabilities? = nil
  let onDone: () -> Void
  private var metadata: String {
    (attachment.pages.map { "\($0) \($0 == 1 ? "page" : "pages") · " } ?? "")
      + ByteCountFormatter.string(fromByteCount: Int64(attachment.size), countStyle: .file)
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      StudioPopoverHeading(title: attachment.name, subtitle: metadata)
        .padding(.horizontal, 9).padding(.top, 8).padding(.bottom, 12)
      if attachment.isPDF {
        StudioPopoverChoice(
          title: "Automatic",
          subtitle: canRasterPDF ? "Read pages as images" : "Images when supported; otherwise text",
          selected: !attachment.textOnly
        ) { attachment.textOnly = false }
        .accessibilityIdentifier("attachment-pdf-automatic")
        StudioPopoverChoice(
          title: "Text only", subtitle: "Extract text, without images or layout",
          selected: attachment.textOnly
        ) { attachment.textOnly = true }
        .accessibilityIdentifier("attachment-pdf-text")
      }
      if attachment.mime.hasPrefix("image/") {
        ForEach(StudioAttachment.imageDetailOptions, id: \.value) { option in
          let estimates = capabilities?.imageEstimates(for: attachment, detail: option.value) ?? []
          let overflow = estimates.first(where: \.exceedsContext)
          StudioPopoverChoice(
            title: option.title,
            subtitle: estimates.isEmpty
              ? nil
              : [
                StudioImageEstimate.label(estimates),
                overflow.map { "Exceeds \($0.modelName)'s context" },
              ]
              .compactMap { $0 }.joined(separator: " · "),
            selected: attachment.detail == option.value
          ) { attachment.detail = option.value }
          .disabled(overflow != nil)
          .help(
            ([option.help] + estimates.map { "\($0.modelName): ≈ \($0.tokens.formatted()) tokens" })
              .joined(separator: "\n")
          )
          .accessibilityIdentifier("attachment-image-\(option.value)")
        }
      }
      if attachment.supportsPageSelection {
        WorkspaceRule().padding(.vertical, 8).padding(.horizontal, 9)
        StudioPopoverChoice(
          title: attachment.pages.map { "All \($0) \($0 == 1 ? "page" : "pages")" } ?? "All pages",
          selected: attachment.allPages
        ) {
          attachment.from = nil
          attachment.to = nil
        }.accessibilityIdentifier("attachment-all-pages")
        HStack(spacing: 8) {
          Text("Range").foregroundStyle(.secondary)
          Spacer(minLength: 8)
          pageField(
            "From page", hint: "1", value: $attachment.firstPage, id: "attachment-page-from")
          Text("-").foregroundStyle(.tertiary)
          pageField(
            "Through page", hint: attachment.pages.map(String.init) ?? "End",
            value: $attachment.lastPage, id: "attachment-page-to")
        }.padding(.horizontal, 9).padding(.vertical, 6)
          .help("Includes both ends. A blank bound includes the first or last page.")
        if let error = attachment.selectionError {
          Text(error).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 9).padding(.top, 4)
            .accessibilityIdentifier("attachment-page-help")
        } else if attachment.pages == nil {
          Text("Page count unavailable").font(.system(size: 11)).foregroundStyle(.secondary)
            .padding(.horizontal, 9).padding(.top, 4)
            .help("The server will validate the range when sent.")
        }
      }
      HStack {
        Spacer()
        Button("Done", action: onDone)
          .buttonStyle(QuietButtonStyle()).font(.system(size: 12, weight: .medium))
          .padding(.horizontal, 9).padding(.vertical, 7)
          .disabled(attachment.selectionError != nil)
      }.padding(.top, 4)
    }.padding(7).frame(width: 290).studioPopoverSurface()
  }
  private func pageField(_ title: String, hint: String, value: Binding<String>, id: String)
    -> some View
  {
    TextField(hint, text: value).textFieldStyle(StudioPopoverFieldStyle())
      .monospacedDigit().multilineTextAlignment(.center).frame(width: 64)
      .accessibilityLabel(title).accessibilityIdentifier(id)
  }
}
