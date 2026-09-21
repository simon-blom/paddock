import AppKit
import SwiftUI

/// A menu is a whole button, not just the painted dots. The plain menu style
/// exposed a 12×2-point glyph as its hit target. Keep the native menu while
/// explicitly sizing its label and interaction region, including empty padding.
struct StudioHistoryActionsMenu<Content: View>: View {
  let title: String
  let id: String
  @ViewBuilder var content: Content

  var body: some View {
    Menu {
      content
    } label: {
      Image(systemName: "ellipsis").font(.system(size: 13))
        .frame(width: 28, height: 28).contentShape(Rectangle())
    }.menuStyle(.button).buttonStyle(QuietButtonStyle()).menuIndicator(.hidden)
      .fixedSize().help("Conversation actions")
      .accessibilityLabel("Actions for \(title)")
      .accessibilityIdentifier("chat-actions-\(id)")
  }
}

/// The title is stored metadata, not a newly generated summary. Reading it
/// requires neither opening/hydrating the conversation nor calling a model.
struct StudioFullChatTitle: View {
  let title: String
  @State private var copied = false
  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      StudioPopoverHeading(title: "Conversation title")
      PaddockScrollView {
        Text(verbatim: title).textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
          .frame(maxWidth: .infinity, alignment: .leading)
          .accessibilityIdentifier("chat-full-title")
      }.frame(maxHeight: 240).fixedSize(horizontal: false, vertical: true)
      HStack {
        Spacer()
        Button(copied ? "Copied" : "Copy title", systemImage: copied ? "checkmark" : "doc.on.doc") {
          NSPasteboard.general.clearContents()
          copied = NSPasteboard.general.setString(title, forType: .string)
        }.buttonStyle(QuietButtonStyle()).accessibilityIdentifier("chat-copy-title")
      }
    }.padding(16).frame(width: 340).studioPopoverSurface()
  }
}
