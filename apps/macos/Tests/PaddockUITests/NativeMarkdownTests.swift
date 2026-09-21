import AppKit
import SwiftUI
import Testing

@testable import PaddockNativeMarkdown

@Suite("Native Markdown trial", .serialized)
struct NativeMarkdownTests {
  @Test func syntaxDecorationPreservesExactSourceInBothThemes() async throws {
    for dark in [false, true] {
      for code in [
        "let text = \"Hello 🦊\"\n\tprint(text)\n", "", "// <tag> & **literal**\nlet x = 42",
      ] {
        let value = try await NativeCodeHighlighter.shared.highlight(
          code, language: "swift", dark: dark)
        #expect(String(value.characters) == code)
      }
    }
  }

  @Test func supportedDiagramsLayoutOffMain() async throws {
    for source in [
      "flowchart LR\n A[Start] --> B[End]",
      "stateDiagram-v2\n [*] --> Ready\n Ready --> Done",
      "sequenceDiagram\n Alice->>Bob: Hello\n Bob-->>Alice: Hi",
      "classDiagram\n Animal <|-- Duck",
      "erDiagram\n CUSTOMER ||--o{ ORDER : places",
      "xychart-beta\n x-axis [A, B, C]\n y-axis 0 --> 10\n bar [3, 7, 5]",
    ] {
      let result = try await MermaidLayoutWorker.shared.layout(source)
      #expect(result.width > 0 && result.height > 0)
    }
  }
  @Test func excessiveDiagramAndUnsupportedSyntaxFailExplicitly() async {
    await #expect(throws: (any Error).self) {
      try await MermaidLayoutWorker.shared.layout(String(repeating: "A --> B;", count: 300))
    }
    await #expect(throws: (any Error).self) {
      try await MermaidLayoutWorker.shared.layout("pie title Chart\n \"A\" : 50")
    }
  }
  @Test @MainActor func nativeViewMountsCodeMathAndMermaid() async throws {
    _ = NSApplication.shared
    #expect(!NativeDiagramView().isFlipped)
    let view = NSHostingView(
      rootView: NativeMarkdown(NativeMarkdownFixtures.showcase).frame(width: 760))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 820, height: 800), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = view
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    func hasDiagram(_ v: NSView) -> Bool {
      v is NativeDiagramView || v.subviews.contains(where: hasDiagram)
    }
    for _ in 0..<100 {
      view.layoutSubtreeIfNeeded()
      if hasDiagram(view) { break }
      try await Task.sleep(for: .milliseconds(25))
    }
    #expect(view.fittingSize.height > 500)
    #expect(hasDiagram(view))
  }

  @Test @MainActor func continuousSelectionSpansWrappedLinesHeadingsAndLists() async throws {
    let markdown = """
      ## Selection heading

      First paragraph has **bold text** and enough ordinary words to wrap across several lines in the narrow fixture window.

      ### Second heading

      Second paragraph ends here.

      - First list row
      - Second list row
      """
    let (host, window) = mount(markdown, width: 340)
    defer { window.close() }
    let textView = try await textSurface(in: host, containing: "Second list row")
    #expect(textView.isSelectable)
    #expect(!textView.isEditable)
    let text = textView.string as NSString
    let start = text.range(of: "paragraph has").location
    let end = NSMaxRange(text.range(of: "Second list row"))
    textView.setSelectedRange(NSRange(location: start, length: end - start))
    #expect(textView.selectedRange().length == end - start)
    let copied = try copySelection(textView)
    #expect(copied.contains("bold text"))
    #expect(copied.contains("Second heading"))
    #expect(copied.contains("Second paragraph"))
    #expect(copied.contains("First list row"))
    #expect(copied.hasSuffix("Second list row"))
    #expect(!copied.contains("**"))
    let selected = textView.selectedRanges
    host.rootView = SelectionFixture(text: markdown, streaming: false, width: 520)
    try await Task.sleep(for: .milliseconds(100))
    host.layoutSubtreeIfNeeded()
    #expect(textView.selectedRanges == selected)
    #expect(try copySelection(textView) == copied)
    let bodyFont = textView.textStorage?.attribute(.font, at: start, effectiveRange: nil) as? NSFont
    #expect(bodyFont?.pointSize == 15)
  }

  @Test @MainActor func selectionCopiesAcrossEmbeddedCodeTableAndDiagram() async throws {
    let markdown = """
      Before the attachments.

      ```swift
      let answer = 42
      ```

      | Name | Value |
      | --- | --- |
      | Sample | forty-two |

      ```mermaid
      flowchart LR
        A[Start] --> B[End]
      ```

      After the attachments.
      """
    let (host, window) = mount(markdown)
    defer { window.close() }
    let textView = try await textSurface(in: host, containing: "After the attachments.")
    textView.setSelectedRange(NSRange(location: 0, length: (textView.string as NSString).length))
    let copied = try copySelection(textView)
    #expect(copied.contains("Before the attachments."))
    #expect(copied.contains("let answer = 42"))
    #expect(copied.contains("forty-two"))
    #expect(copied.contains("A[Start] --> B[End]"))
    #expect(copied.contains("After the attachments."))
    #expect(!copied.contains("\u{FFFC}"))
  }

  @Test @MainActor func streamingPreservesAnExistingSelection() async throws {
    let prefix = "First **selected** paragraph.\n\nSecond selected paragraph.\n\n"
    let (host, window) = mount(prefix + "Still *stream", streaming: true)
    defer { window.close() }
    let textView = try await textSurface(in: host, containing: "Second selected paragraph.")
    let selected = NSRange(
      location: 0,
      length: NSMaxRange((textView.string as NSString).range(of: "Second selected paragraph.")))
    textView.setSelectedRange(selected)
    let original = try copySelection(textView)
    for (suffix, streaming) in [
      ("Still *streaming with more tokens", true),
      ("Still *streaming with more tokens*.\n\nFinished.", false),
    ] {
      host.rootView = SelectionFixture(text: prefix + suffix, streaming: streaming, width: 760)
      let updated = try await textSurface(
        in: host, containing: streaming ? "more tokens" : "Finished.")
      #expect(updated === textView)
      #expect(updated.selectedRange() == selected)
      #expect(try copySelection(updated) == original)
    }
  }

  @Test @MainActor func rewrittenTextDoesNotRestoreAnInvalidSelection() async throws {
    let (host, window) = mount("First paragraph.\n\nSelect this old paragraph.")
    defer { window.close() }
    let textView = try await textSurface(in: host, containing: "Select this old paragraph.")
    textView.setSelectedRange((textView.string as NSString).range(of: "Select this old paragraph."))
    host.rootView = SelectionFixture(text: "A new and shorter reply.", streaming: false, width: 760)
    let updated = try await textSurface(in: host, containing: "A new and shorter reply.")
    #expect(updated.selectedRange().length == 0)
  }

  private struct SelectionFixture: View {
    let text: String
    let streaming: Bool
    let width: CGFloat
    var body: some View { NativeMarkdown(text, streaming: streaming).frame(width: width) }
  }

  @MainActor private func mount(_ text: String, streaming: Bool = false, width: CGFloat = 760) -> (
    NSHostingView<SelectionFixture>, NSWindow
  ) {
    _ = NSApplication.shared
    let host = NSHostingView(
      rootView: SelectionFixture(text: text, streaming: streaming, width: width))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: width, height: 800), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    return (host, window)
  }

  @MainActor private func textSurface(in view: NSView, containing marker: String) async throws
    -> NSTextView
  {
    func find(_ view: NSView) -> NSTextView? {
      if let text = view as? NSTextView, text.string.contains(marker) { return text }
      return view.subviews.lazy.compactMap { find($0) }.first
    }
    for _ in 0..<100 {
      view.layoutSubtreeIfNeeded()
      if let found = find(view) { return found }
      try await Task.sleep(for: .milliseconds(25))
    }
    return try #require(find(view), "No continuous text surface contains \(marker)")
  }

  @MainActor private func copySelection(_ view: NSTextView) throws -> String {
    // Exercise AppKit's real copy path without modifying the user's clipboard.
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    #expect(view.window?.makeFirstResponder(view) == true)
    #expect(view.selectedRange().length > 0)
    #expect(view.writeSelection(to: pasteboard, types: view.writablePasteboardTypes))
    return try #require(pasteboard.string(forType: .string))
  }
}
