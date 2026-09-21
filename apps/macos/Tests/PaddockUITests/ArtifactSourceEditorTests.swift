import AppKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native artifact source syntax")
struct ArtifactSourceSyntaxTests {
  @Test func embeddedHTMLAndMultilineState() throws {
    let text =
      "<style>\n.hero { color: red; /* start\nend */ padding: 12px; }\n</style>\n<script>\nconst title = `Hello\nworld`;\n</script>\n<div class=\"hero\">🦊</div>\n"
    let parsed = try ArtifactSourceSyntax.parse(text, language: "html")
    #expect(parsed.lines.count == 10)
    #expect(parsed.lines[1].tokens.contains { $0.kind == .attribute })
    #expect(parsed.lines[2].tokens.first?.kind == .comment)
    #expect(parsed.lines[5].tokens.first?.kind == .keyword)
    #expect(parsed.lines[6].tokens.first?.kind == .string)
    #expect(parsed.lines[8].tokens.contains { $0.kind == .tag })
    #expect(parsed.lines[8].tokens.contains { $0.kind == .attribute })
    #expect(parsed.lines.last?.output.embedded == "")
    for i in parsed.lines.indices {
      for token in parsed.lines[i].tokens {
        #expect(NSMaxRange(token.range) <= parsed.lines[i].text.utf16.count)
      }
    }
  }

  @Test func editsReuseSuffixButPropagateCommentChanges() throws {
    let text = (0..<5000).map { "const value\($0) = \($0);\n" }.joined()
    let baseline = try ArtifactSourceSyntax.parse(text, language: "js")
    let edited = text.replacingOccurrences(of: "value2500", with: "renamed2500")
    let next = try ArtifactSourceSyntax.parse(edited, language: "js", previous: baseline)
    #expect(next.lexedLines == 1)
    let commented = try ArtifactSourceSyntax.parse(
      "/*\n" + text, language: "js", previous: baseline)
    #expect(commented.lines[2500].tokens.first?.kind == .comment)
    let restored = try ArtifactSourceSyntax.parse(text, language: "js", previous: commented)
    #expect(restored.lines[2500].tokens.first?.kind == .keyword)
  }

  @Test func rawTextEndTagsAndFencedLanguages() throws {
    let html = try ArtifactSourceSyntax.parse(
      "<SCRIPT>const text = `hello </SCRIPT><h1>End</h1>", language: "html")
    #expect(html.lines[0].output.embedded.isEmpty)
    #expect(html.lines[0].output.quote == 0)
    #expect(html.lines[0].tokens.filter { $0.kind == .tag }.count >= 6)
    let markdown = try ArtifactSourceSyntax.parse(
      "# Title\n```js\nconst text = `hello\n```\nplain words\n", language: "markdown")
    #expect(markdown.lines[2].tokens.first?.kind == .keyword)
    #expect(markdown.lines[4].tokens.isEmpty)
    let python = try ArtifactSourceSyntax.parse(
      "value = \"\"\"first\nsecond\"\"\"\nreturn value\n", language: "python")
    #expect(python.lines[1].tokens.first?.kind == .string)
    #expect(python.lines[2].tokens.first?.kind == .keyword)
  }

  @Test func lineMapHandlesUnicodeCRLFAndTrailingEmptyLine() throws {
    let parsed = try ArtifactSourceSyntax.parse("🦊é\r\nβ\r\n", language: "text")
    #expect(parsed.starts == [0, 6, 9])
    #expect(parsed.line(at: 5) == 0)
    #expect(parsed.line(at: 6) == 1)
    #expect(parsed.line(at: 9) == 2)
    #expect(try ArtifactSourceSyntax.parse("", language: "text").starts == [0])
  }

  @Test func cancellationAndLargeFixtures() async throws {
    let text = String(repeating: "<div class=\"row\">Hello 🦊</div>\n", count: 50000)
    let clock = ContinuousClock()
    let started = clock.now
    let result = try await Task.detached { try ArtifactSourceSyntax.parse(text, language: "html") }
      .value
    print(
      "Source syntax: \(text.utf8.count) bytes / \(result.lines.count) lines: \(started.duration(to: clock.now))"
    )
    #expect(result.lines.count == 50001 && result.length == text.utf16.count)
    let minified = String(repeating: "const x=123;", count: 100000)
    let long = try await Task.detached { try ArtifactSourceSyntax.parse(minified, language: "js") }
      .value
    #expect(long.length == minified.utf16.count && long.lines.count == 1)
    let cancelled = Task.detached {
      withUnsafeCurrentTask { $0?.cancel() }
      return try ArtifactSourceSyntax.parse(text, language: "html")
    }
    await #expect(throws: CancellationError.self) { try await cancelled.value }
  }
}

@Suite("Native artifact source editor", .serialized) @MainActor
struct ArtifactSourceEditorTests {
  private func session(
    _ text: String = "<h1>Hello</h1>\n", readOnly: Bool = false, dark: Bool = false
  ) -> ArtifactSourceSession {
    _ = NSApplication.shared
    let result = ArtifactSourceSession()
    result.update(identity: "fixture", text: text, language: "html", readOnly: readOnly, dark: dark)
    return result
  }
  private func window(_ session: ArtifactSourceSession) -> NSWindow {
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 600, height: 400), styleMask: [.borderless],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = session.scrollView
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    window.makeFirstResponder(session.textView)
    return window
  }
  private func settled(_ session: ArtifactSourceSession) async throws {
    for _ in 0..<500 {
      if session.appliedRevision == session.revision { return }
      try await Task.sleep(for: .milliseconds(10))
    }
    Issue.record("Syntax worker did not settle")
  }

  @Test func nativeTextSystemSelectionUndoAndBindingEcho() async throws {
    let editor = session("one\ntwo 🦊\nthree")
    let window = window(editor)
    defer {
      editor.suspend()
      window.close()
    }
    var changed = ""
    editor.onChange = { changed = $0 }
    editor.textView.selectAll(nil)
    #expect(editor.textView.selectedRange().length == editor.textView.string.utf16.count)
    editor.textView.setSelectedRange(NSRange(location: 0, length: 0))
    editor.textView.insertText("X", replacementRange: editor.textView.selectedRange())
    #expect(changed.hasPrefix("Xone"))
    let selection = editor.textView.selectedRange()
    editor.update(identity: "fixture", text: changed, language: "html", readOnly: false, dark: true)
    #expect(editor.textView.selectedRange() == selection)
    #expect(editor.textView.undoManager?.canUndo == true)
    try await settled(editor)
    #expect(editor.textView.textLayoutManager != nil)
    #expect(editor.scrollView.clipsToBounds)
    #expect(
      editor.scrollView.verticalRulerView?.clipsToBounds == true,
      "The gutter must not paint over source or the SwiftUI artifact header")
    editor.textView.undoManager?.undo()
    #expect(editor.textView.string == "one\ntwo 🦊\nthree")
    editor.textView.undoManager?.redo()
    #expect(editor.textView.string.hasPrefix("Xone"))
  }

  @Test func indentationNewlinesAndUndoAreSingleTransactions() {
    let editor = session("one\r\ntwo\r\nthree")
    let window = window(editor)
    defer {
      editor.suspend()
      window.close()
    }
    editor.textView.setSelectedRange(NSRange(location: 0, length: 10))
    editor.textView.insertTab(nil)
    #expect(editor.textView.string == "  one\r\n  two\r\nthree")
    editor.textView.undoManager?.undo()
    #expect(editor.textView.string == "one\r\ntwo\r\nthree")
    editor.textView.setSelectedRange(NSRange(location: 0, length: 10))
    editor.textView.insertTab(nil)
    editor.textView.insertBacktab(nil)
    #expect(editor.textView.string == "one\r\ntwo\r\nthree")
    editor.update(
      identity: "next", text: "  if (x) {\r\n}", language: "js", readOnly: false, dark: false)
    editor.textView.setSelectedRange(NSRange(location: 10, length: 0))
    editor.textView.insertNewline(nil)
    #expect(editor.textView.string == "  if (x) {\r\n    \r\n}")
  }

  @Test func previewRoundtripPreservesSelectionUndoAndVersionIsReadOnly() async throws {
    let editor = session()
    let window = window(editor)
    defer {
      editor.suspend()
      window.close()
    }
    editor.textView.insertText("X", replacementRange: NSRange(location: 4, length: 0))
    let value = editor.textView.string
    let selection = editor.textView.selectedRange()
    editor.suspend()
    editor.update(identity: "fixture", text: value, language: "html", readOnly: false, dark: false)
    #expect(
      editor.textView.selectedRange() == selection && editor.textView.undoManager?.canUndo == true)
    editor.update(identity: "v1", text: "original", language: "html", readOnly: true, dark: false)
    #expect(!editor.textView.isEditable && editor.textView.isSelectable)
    #expect(editor.textView.undoManager?.canUndo == false)
    try await settled(editor)
    #expect(editor.syntax?.length == 8)
  }

  @Test func staleWorkersThemeChangesAndNativeFind() async throws {
    let editor = session(String(repeating: "let x = 1\n", count: 50000))
    let window = window(editor)
    defer {
      editor.suspend()
      window.close()
    }
    editor.update(
      identity: "fixture", text: "let final = 42\n", language: "js", readOnly: false, dark: true)
    try await settled(editor)
    #expect(editor.syntax?.length == 15)
    #expect(editor.syntax?.language == "javascript")
    let event = try #require(
      NSEvent.keyEvent(
        with: .keyDown, location: .zero, modifierFlags: .command, timestamp: 1,
        windowNumber: window.windowNumber, context: nil, characters: "f",
        charactersIgnoringModifiers: "f", isARepeat: false, keyCode: 3))
    #expect(editor.textView.performKeyEquivalent(with: event))
    #expect(editor.scrollView.isFindBarVisible)
    #expect(editor.textView.textLayoutManager != nil)
  }

  @Test func saveShortcutOnlyTargetsFocusedComparePane() throws {
    let a = session()
    let b = session()
    let window = window(a)
    defer {
      a.suspend()
      b.suspend()
      window.close()
    }
    var savedA = 0
    var savedB = 0
    a.textView.save = { savedA += 1 }
    b.textView.save = { savedB += 1 }
    let event = try #require(
      NSEvent.keyEvent(
        with: .keyDown, location: .zero, modifierFlags: .command, timestamp: 1,
        windowNumber: window.windowNumber, context: nil, characters: "s",
        charactersIgnoringModifiers: "s", isARepeat: false, keyCode: 1))
    #expect(a.textView.performKeyEquivalent(with: event))
    _ = b.textView.performKeyEquivalent(with: event)
    #expect(savedA == 1 && savedB == 0)
  }

  @Test func largeDocumentViewportEditingAndLifetime() async throws {
    let text = String(repeating: "<div class=\"row\">Hello 🦊</div>\n", count: 50000)
    let clock = ContinuousClock()
    let start = clock.now
    var editor: ArtifactSourceSession? = session(text)
    let loaded = start.duration(to: clock.now)
    weak let released = editor
    let window = window(editor!)
    try await settled(editor!)
    window.contentView?.layoutSubtreeIfNeeded()
    window.contentView?.displayIfNeeded()
    let initial = start.duration(to: clock.now)
    var samples: [Double] = []
    for _ in 0..<20 {
      let time = clock.now
      editor!.textView.insertText("x", replacementRange: NSRange(location: 5, length: 0))
      window.contentView?.layoutSubtreeIfNeeded()
      window.contentView?.displayIfNeeded()
      let elapsed = time.duration(to: clock.now).components
      samples.append(Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15)
    }
    try await settled(editor!)
    #expect(editor!.syntax?.lexedLines == 1)
    #expect(editor!.textView.textLayoutManager != nil)
    #expect(editor!.textView.string.utf16.count == text.utf16.count + 20)
    print(
      "Source viewport 50k lines: assignment=\(loaded), first draw+syntax=\(initial), edit+draw max=\(samples.max()!)ms, median=\(samples.sorted()[10])ms"
    )
    editor!.suspend()
    window.contentView = nil
    window.close()
    editor = nil
    try await Task.sleep(for: .milliseconds(80))
    #expect(released == nil, "Closing an artifact must release its editor, tokens and text storage")
  }

  @Test func minifiedSourceRetainsContentAndWrapsOnResize() async throws {
    let text = String(repeating: "const x=123;", count: 100000)
    let editor = session(text)
    let clock = ContinuousClock()
    let start = clock.now
    let window = window(editor)
    defer {
      editor.suspend()
      window.close()
    }
    try await settled(editor)
    window.contentView?.layoutSubtreeIfNeeded()
    window.contentView?.displayIfNeeded()
    let first = start.duration(to: clock.now)
    let resize = clock.now
    window.setContentSize(NSSize(width: 360, height: 400))
    window.contentView?.layoutSubtreeIfNeeded()
    window.contentView?.displayIfNeeded()
    #expect(editor.textView.string == text)
    #expect(editor.textView.textContainer?.widthTracksTextView == true)
    #expect(editor.textView.textLayoutManager != nil)
    print(
      "Source minified 1.1MB: first draw+syntax=\(first), resize+draw=\(resize.duration(to: clock.now))"
    )
  }

  @Test func sourceAndHeaderRenderInBothAppearances() async throws {
    let text =
      "<!DOCTYPE html>\n<html lang=\"en\">\n<style>\n  .hero { color: red; padding: 12px; }\n</style>\n<script>\n  const title = `Hello`;\n</script>\n<h1>Native artifact editor</h1>\n</html>"
    for dark in [false, true] {
      let editor = ArtifactSourceSession()
      let host = NSHostingController(
        rootView: VStack(spacing: 0) {
          Text("Artifact source").frame(maxWidth: .infinity).frame(height: 36)
          ArtifactSourceEditor(
            session: editor, identity: "fixture", text: .constant(text), language: "html",
            readOnly: false, save: {})
        }.environment(\.colorScheme, dark ? .dark : .light))
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 600, height: 400),
        styleMask: [.borderless], backing: .buffered, defer: false)
      host.sizingOptions = []
      window.isReleasedWhenClosed = false
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      window.contentViewController = host
      window.setContentSize(NSSize(width: 600, height: 400))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer {
        editor.suspend()
        window.close()
      }
      host.view.layoutSubtreeIfNeeded()
      try await settled(editor)
      host.view.layoutSubtreeIfNeeded()
      host.view.displayIfNeeded()
      #expect(editor.textView.string == text)
      #expect(editor.textView.visibleRect.width > 500)
      #expect(editor.scrollView.frame.height <= 365)
      let bitmap = try #require(host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds))
      host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
      // Exclude gutter/header. Syntax must actually paint colored glyphs, not
      // merely exist in accessibility or the text storage under an opaque view.
      var colored = 0
      let scale = CGFloat(bitmap.pixelsWide) / host.view.bounds.width
      for y in Int(44 * scale)..<min(bitmap.pixelsHigh, Int(240 * scale)) {
        for x in Int(50 * scale)..<min(bitmap.pixelsWide, Int(500 * scale)) {
          if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
            max(color.redComponent, color.greenComponent, color.blueComponent)
              - min(color.redComponent, color.greenComponent, color.blueComponent) > 0.12
          {
            colored += 1
          }
        }
      }
      #expect(
        colored > 100, "Syntax glyphs must remain visible in \(dark ? "dark" : "light") appearance")
      let long = String(repeating: "<div class=\"row\">content</div>\n", count: 1000)
      editor.update(identity: "fixture", text: long, language: "html", readOnly: false, dark: dark)
      try await settled(editor)
      editor.textView.scrollRangeToVisible(NSRange(location: long.utf16.count - 20, length: 1))
      try await Task.sleep(for: .milliseconds(80))
      host.view.layoutSubtreeIfNeeded()
      host.view.displayIfNeeded()
      let manager = try #require(editor.textView.textLayoutManager)
      let viewport = try #require(manager.textViewportLayoutController.viewportRange)
      let storage = try #require(manager.textContentManager)
      #expect(storage.offset(from: storage.documentRange.location, to: viewport.location) > 10000)
      var hasVisibleColor = false
      manager.enumerateRenderingAttributes(from: viewport.location, reverse: false) {
        _, attributes, range in
        if range.location.compare(viewport.endLocation) != .orderedAscending { return false }
        if let color = (attributes[.foregroundColor] as? NSColor)?.usingColorSpace(.deviceRGB),
          abs(color.redComponent - color.blueComponent) > 0.12
        {
          hasVisibleColor = true
        }
        return true
      }
      #expect(hasVisibleColor, "Scrolling to previously unpainted source must apply syntax colors")
      if ProcessInfo.processInfo.environment["PADDOCK_EDITOR_CAPTURE"] == "1" {
        try bitmap.representation(using: .png, properties: [:])?.write(
          to: URL(fileURLWithPath: "/tmp/paddock-source-editor-\(dark ? "dark" : "light").png"))
      }
    }
  }
}
