import AppKit
import SwiftUI
import Testing

@testable import PaddockNativeMarkdown

@Suite("Native OCR confidence decoration", .serialized)
struct NativeConfidenceTests {
  @Test func literalUnicodeMatchesAreBoundedAndSkipExcludedCode() {
    let text = "Ångström **price** 🦊 Ångström price a+b code"
    let ranges = NativeConfidenceMatches.ranges(
      text: text, words: ["Ångström", "price", "a+b", "🦊", "code"],
      excluding: [(text as NSString).range(of: "code")])
    #expect(
      ranges.map { (text as NSString).substring(with: $0) } == [
        "Ångström", "price", "Ångström", "price", "a+b",
      ])
    #expect(
      NativeConfidenceMatches.ranges(
        text: String(repeating: "word ", count: 10000), words: ["word"]
      ).count == 4096)
    #expect(NativeConfidenceMatches.ranges(text: "short", words: []).isEmpty)
  }
  @Test @MainActor func decorationDoesNotModifyStorageSelectionOrCopyInEitherTextKit() async throws
  {
    for modern in [false, true] {
      for dark in [false, true] {
        let text = NSTextView(usingTextLayoutManager: modern)
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: 600, height: 300), styleMask: [.titled],
          backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = text
        defer { window.close() }
        text.isSelectable = true
        text.isEditable = false
        #expect(window.makeFirstResponder(text))
        text.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        let original = NSMutableAttributedString(
          string: "Before unsure after. `unsure` code unsure",
          attributes: [.font: NSFont.systemFont(ofSize: 15)])
        let inline = (original.string as NSString).range(of: "`unsure`")
        original.addAttribute(.backgroundColor, value: NSColor.quaternaryLabelColor, range: inline)
        let code = (original.string as NSString).range(of: "code unsure")
        original.addAttribute(
          .font, value: NSFont.monospacedSystemFont(ofSize: 15, weight: .regular), range: code)
        text.textStorage?.setAttributedString(original)
        text.setSelectedRange(NSRange(location: 0, length: 19))
        let decoration = NativeConfidenceDecoration()
        decoration.words = ["unsure"]
        decoration.reconcile(text)
        for _ in 0..<100 {
          if !marked(text).isEmpty { break }
          try await Task.sleep(for: .milliseconds(10))
        }
        #expect(marked(text) == [(original.string as NSString).range(of: "unsure")])
        #expect(text.textStorage?.isEqual(to: original) == true)
        #expect(text.selectedRange() == NSRange(location: 0, length: 19))
        let board = NSPasteboard.withUniqueName()
        defer { board.releaseGlobally() }
        #expect(text.writeSelection(to: board, types: text.writablePasteboardTypes))
        #expect(board.string(forType: .string) == "Before unsure after")
        decoration.words = []
        for _ in 0..<100 {
          if marked(text).isEmpty { break }
          try await Task.sleep(for: .milliseconds(10))
        }
        #expect(marked(text).isEmpty)
        #expect(text.textStorage?.isEqual(to: original) == true)
      }
    }
  }
  @Test @MainActor func actualMarkdownMarksProseButNotInlineOrFencedCode() async throws {
    _ = NSApplication.shared
    let markdown =
      "# Document\n\nAn **unsure** total.\n\nKeep `unsure` unchanged.\n\n```text\nunsure\n```\n\nEnd."
    let host = NSHostingView(
      rootView: NativeMarkdown(markdown, unsureWords: ["unsure"]).frame(width: 600))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 640, height: 800), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    defer { window.close() }
    func surfaces(_ view: NSView) -> [NSTextView] {
      (view as? NSTextView).map { [$0] } ?? view.subviews.flatMap(surfaces)
    }
    var all: [NSTextView] = []
    for _ in 0..<100 {
      host.layoutSubtreeIfNeeded()
      all = surfaces(host)
      if all.reduce(0, { $0 + marked($1).count }) == 1 { break }
      try await Task.sleep(for: .milliseconds(25))
    }
    #expect(all.reduce(0, { $0 + marked($1).count }) == 1)
    #expect(all.contains { $0.string.contains("Keep unsure unchanged.") })
    let prose = try #require(all.first { $0.string.contains("An unsure total.") })
    #expect(marked(prose) == [(prose.string as NSString).range(of: "unsure")])
  }

  @MainActor private func marked(_ view: NSTextView) -> [NSRange] {
    var ranges: [NSRange] = []
    if let layout = view.textLayoutManager, let content = layout.textContentManager {
      layout.enumerateRenderingAttributes(from: content.documentRange.location, reverse: false) {
        _, attrs, range in
        if attrs[.backgroundColor] != nil {
          ranges.append(
            NSRange(
              location: content.offset(from: content.documentRange.location, to: range.location),
              length: content.offset(from: range.location, to: range.endLocation)))
        }
        return true
      }
    } else if let layout = view.layoutManager {
      var offset = 0
      while offset < (view.string as NSString).length {
        var range = NSRange()
        if layout.temporaryAttribute(
          .backgroundColor, atCharacterIndex: offset, effectiveRange: &range) != nil
        {
          ranges.append(range)
        }
        offset = max(offset + 1, NSMaxRange(range))
      }
    }
    return ranges
  }
}
