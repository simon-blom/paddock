import AppKit
import Testing

@testable import PaddockUI

@Suite("Smart native source editing", .serialized) @MainActor
struct ArtifactSmartEditingTests {
  private func editor(_ text: String = "", language: String = "javascript", readOnly: Bool = false)
    -> ArtifactSourceSession
  {
    _ = NSApplication.shared
    let editor = ArtifactSourceSession()
    editor.update(
      identity: "smart", text: text, language: language, readOnly: readOnly, dark: false)
    return editor
  }
  private func type(_ value: String, in view: ArtifactCodeTextView) {
    view.insertText(value, replacementRange: NSRange(location: NSNotFound, length: 0))
  }
  private func settled(_ editor: ArtifactSourceSession) async throws {
    for _ in 0..<500 {
      if editor.revision == editor.appliedRevision { return }
      try await Task.sleep(for: .milliseconds(10))
    }
    Issue.record("Source analysis did not settle")
  }

  @Test func nestedPairInsertionOvertypeAndDeletion() {
    let session = editor()
    defer { session.suspend() }
    let view = session.textView
    type("(", in: view)
    #expect(view.string == "()")
    type("[", in: view)
    #expect(view.string == "([])")
    type("]", in: view)
    type(")", in: view)
    #expect(view.string == "([])" && view.selectedRange().location == 4)
    session.update(identity: "empty", text: "", language: "js", readOnly: false, dark: false)
    type("{", in: view)
    view.deleteBackward(nil)
    #expect(view.string.isEmpty)
    view.undoManager?.undo()
    #expect(view.string == "{}")
    view.undoManager?.undo()
    #expect(view.string.isEmpty)
  }

  @Test func onlyOwnedClosersAreOvertypedAndMarkersTrackOtherEdits() {
    let session = editor(")")
    defer { session.suspend() }
    let view = session.textView
    type(")", in: view)
    #expect(view.string == "))")
    session.update(identity: "new", text: "", language: "js", readOnly: false, dark: false)
    type("(", in: view)
    type("hello", in: view)
    type(")", in: view)
    #expect(view.string == "(hello)" && view.selectedRange().location == 7)
    session.update(identity: "new2", text: "", language: "js", readOnly: false, dark: false)
    type("(", in: view)
    view.setSelectedRange(NSRange(location: 0, length: 0))
    type("x", in: view)
    view.setSelectedRange(NSRange(location: 2, length: 0))
    type(")", in: view)
    #expect(view.string == "x()")
  }

  @Test func wrapsUnicodeSelectionAndUndoesInOneStep() {
    let session = editor("🦊é")
    defer { session.suspend() }
    let view = session.textView
    view.selectAll(nil)
    type("[", in: view)
    #expect(view.string == "[🦊é]")
    #expect(view.selectedRange() == NSRange(location: 1, length: 4))
    view.undoManager?.undo()
    #expect(view.string == "🦊é")
    view.undoManager?.redo()
    #expect(view.string == "[🦊é]")
  }

  @Test func literalsCommentsPlainTextAndIdentifiersDoNotGainPairs() {
    for (text, language, typed) in [
      ("// note ", "js", "{"), ("let s = \"hello ", "js", "("), ("# note ", "python", "["),
      ("/* note ", "css", "{"), ("don't", "text", "'"), ("word", "js", "'"),
      ("<p>Hello ", "html", "{"),
    ] {
      let session = editor(text, language: language)
      defer { session.suspend() }
      let view = session.textView
      view.setSelectedRange(NSRange(location: text.utf16.count, length: 0))
      type(typed, in: view)
      #expect(view.string == text + typed, "\(language): \(text)")
    }
    let session = editor("<div class=", language: "html")
    defer { session.suspend() }
    session.textView.setSelectedRange(NSRange(location: 11, length: 0))
    type("\"", in: session.textView)
    #expect(session.textView.string == "<div class=\"\"")
  }

  @Test func pasteAndIMECommitDoNotGainDelimiters() {
    let session = editor()
    defer { session.suspend() }
    // Use the normal insertText multi-character path for pasted source without
    // changing the user's global clipboard in an automated test.
    type("const x = {", in: session.textView)
    #expect(session.textView.string == "const x = {")
    session.textView.performPlainInsertion { type("(", in: session.textView) }
    #expect(
      session.textView.string == "const x = {(",
      "Even a single pasted delimiter must remain literal")
    session.update(identity: "ime", text: "", language: "js", readOnly: false, dark: false)
    session.textView.setMarkedText(
      "(", selectedRange: NSRange(location: 1, length: 0),
      replacementRange: NSRange(location: NSNotFound, length: 0))
    #expect(session.textView.hasMarkedText())
    type("(", in: session.textView)
    #expect(session.textView.string == "(")
    #expect(!session.textView.hasMarkedText())
  }

  @Test func newlineBetweenBracketsPreservesCRLFAndTabs() {
    let session = editor("\t{}\r\n")
    defer { session.suspend() }
    let view = session.textView
    view.setSelectedRange(NSRange(location: 2, length: 0))
    view.insertNewline(nil)
    #expect(view.string == "\t{\r\n\t\t\r\n\t}\r\n")
    #expect(view.selectedRange().location == 6)
    view.undoManager?.undo()
    #expect(view.string == "\t{}\r\n")
    session.update(
      identity: "python", text: "if ready:", language: "python", readOnly: false, dark: false)
    view.setSelectedRange(NSRange(location: 9, length: 0))
    view.insertNewline(nil)
    #expect(view.string == "if ready:\n  ")
    session.update(
      identity: "text", text: "Heading:", language: "text", readOnly: false, dark: false)
    view.setSelectedRange(NSRange(location: 8, length: 0))
    view.insertNewline(nil)
    #expect(view.string == "Heading:\n")
  }

  @Test func tabGoesToNextIndentStop() {
    let session = editor(" x")
    defer { session.suspend() }
    session.textView.setSelectedRange(NSRange(location: 1, length: 0))
    session.textView.insertTab(nil)
    #expect(session.textView.string == "  x")
  }

  @Test func toggleLineCommentsPreservesIndentAndTrailingSelectionBoundary() {
    let session = editor("  one\r\n\ttwo\r\nthree")
    defer { session.suspend() }
    let view = session.textView
    view.setSelectedRange(NSRange(location: 0, length: 13))
    view.toggleSourceComment(nil)
    #expect(view.string == "  // one\r\n\t// two\r\nthree")
    view.toggleSourceComment(nil)
    #expect(view.string == "  one\r\n\ttwo\r\nthree")
    view.undoManager?.undo()
    #expect(view.string == "  // one\r\n\t// two\r\nthree")
    view.undoManager?.undo()
    #expect(view.string == "  one\r\n\ttwo\r\nthree")
  }

  @Test func markupAndCSSCommentsRoundtripAndRejectNestedDelimiters() {
    for (language, original, expected) in [
      ("html", "<h1>Hi</h1>", "<!-- <h1>Hi</h1> -->"),
      ("css", ".hero { color: red; }", "/* .hero { color: red; } */"),
    ] {
      let session = editor(original, language: language)
      defer { session.suspend() }
      session.textView.selectAll(nil)
      session.textView.toggleSourceComment(nil)
      #expect(session.textView.string == expected)
      session.textView.toggleSourceComment(nil)
      #expect(session.textView.string == original)
    }
    let session = editor("body { /* note */ color: red; }", language: "css")
    defer { session.suspend() }
    session.textView.selectAll(nil)
    session.textView.toggleSourceComment(nil)
    #expect(session.textView.string == "body { /* note */ color: red; }")
  }

  @Test func commentPoliciesRespectEmbeddedLanguagesAndStrictJSON() {
    let session = editor("<script>\n  const x = 1;\n</script>", language: "html")
    defer { session.suspend() }
    session.textView.setSelectedRange(NSRange(location: 11, length: 0))
    session.textView.toggleSourceComment(nil)
    #expect(session.textView.string.contains("  // const x = 1;"))
    session.update(
      identity: "json", text: "{\"x\":1}", language: "json", readOnly: false, dark: false)
    session.textView.selectAll(nil)
    session.textView.toggleSourceComment(nil)
    #expect(session.textView.string == "{\"x\":1}")
  }

  @Test func duplicateLastLineAddsSeparatorAndOneUndo() {
    let session = editor("first\r\nlast")
    defer { session.suspend() }
    let view = session.textView
    view.setSelectedRange(NSRange(location: 9, length: 0))
    view.duplicateSourceLines(nil)
    #expect(view.string == "first\r\nlast\r\nlast")
    #expect(view.selectedRange().location == 15)
    view.undoManager?.undo()
    #expect(view.string == "first\r\nlast")
  }

  @Test func bracketIndexIgnoresStringsCommentsAndHighlightsMatchingPair() async throws {
    let session = editor("{\n  const value = [\"}\"]; // ]\n}\n")
    defer { session.suspend() }
    try await settled(session)
    let parsed = try #require(session.syntax)
    #expect(parsed.brackets.count == 4)
    let view = session.textView
    view.setSelectedRange(NSRange(location: 0, length: 0))
    view.jumpToMatchingBracket(nil)
    #expect(view.selectedRange().location == parsed.brackets[0])
    view.jumpToMatchingBracket(nil)
    #expect(view.selectedRange().location == 0)
    type("x", in: view)
    #expect(view.matchAt?(0) == nil, "Never navigate using stale offsets")
  }

  @Test func completionsUseBoundedDocumentVocabularyOutsideLiterals() async throws {
    let session = editor(
      "const heroTitle = 1; const heroSubtitle = 2;\n// secretCommentWord\nconst s = \"secretStringWord\";\nher"
    )
    defer { session.suspend() }
    try await settled(session)
    let view = session.textView
    let length = session.textView.string.utf16.count
    var index = -1
    let result = view.completions(
      forPartialWordRange: NSRange(location: length - 3, length: 3), indexOfSelectedItem: &index)
    #expect(result == ["heroSubtitle", "heroTitle"])
    #expect(
      !view.completionWords.contains("secretCommentWord")
        && !view.completionWords.contains("secretStringWord"))
    #expect(view.completionWords.count <= 8192)
  }

  @Test func readOnlyActionsNeverModifyText() {
    let session = editor("let value = 1", readOnly: true)
    defer { session.suspend() }
    let view = session.textView
    view.selectAll(nil)
    type("(", in: view)
    view.toggleSourceComment(nil)
    view.duplicateSourceLines(nil)
    #expect(view.string == "let value = 1" && view.undoManager?.canUndo == false)
  }

  @Test func rapidTypingAtLargeDocumentEndUsesSafeLocalContext() async throws {
    let text = String(repeating: "const x = 1;\n", count: 10000)
    let session = editor(text)
    defer { session.suspend() }
    try await settled(session)
    let view = session.textView
    view.setSelectedRange(NSRange(location: text.utf16.count, length: 0))
    let clock = ContinuousClock()
    let start = clock.now
    type("(", in: view)
    type("[", in: view)
    type("{", in: view)
    #expect(String(view.string.suffix(6)) == "([{}])")
    type("}", in: view)
    type("]", in: view)
    type(")", in: view)
    #expect(String(view.string.suffix(6)) == "([{}])")
    print("Smart insertion at 130k EOF (6 keys): \(start.duration(to: clock.now))")
    // Changing state much earlier invalidates that checkpoint. Never reuse old
    // end-of-document state to pair a quote inside the newly inserted comment.
    view.setSelectedRange(NSRange(location: 0, length: 0))
    type("/*", in: view)
    view.setSelectedRange(NSRange(location: view.string.utf16.count, length: 0))
    type("'", in: view)
    #expect(String(view.string.suffix(7)) == "([{}])'")
  }

  @Test func codePoliciesAvoidRustLifetimesAndInvalidJSONQuotePairs() {
    for (language, value) in [("rust", "'"), ("swift", "'"), ("json", "'"), ("json", "(")] {
      let session = editor(language: language)
      defer { session.suspend() }
      type(value, in: session.textView)
      #expect(session.textView.string == value)
    }
  }

  @Test func contextMenuIsDiscoverableWithoutAccumulatingItems() throws {
    let session = editor("const x = 1;")
    defer { session.suspend() }
    let view = session.textView
    let event = try #require(
      NSEvent.mouseEvent(
        with: .rightMouseDown, location: .zero, modifierFlags: [], timestamp: 1, windowNumber: 0,
        context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
    for _ in 0..<2 {
      let menu = try #require(view.menu(for: event))
      #expect(menu.items.filter { $0.title == "Toggle Comment" }.count == 1)
      let item = try #require(menu.items.first { $0.title == "Toggle Comment" })
      #expect(view.validateMenuItem(item))
      view.isEditable = false
      #expect(!view.validateMenuItem(item))
      view.isEditable = true
    }
  }

  @Test func sourceAliasesLifetimesAndTripleStringsDoNotConfuseMatching() throws {
    for (source, language) in [
      ("fn f<'a>(x: &'a str) { let c = 'x'; }", "rs"),
      ("let text = \"\"\"\n} ignored\n\"\"\"\nfunc f() {}", "swift"),
    ] {
      let parsed = try ArtifactSourceSyntax.parse(source, language: language)
      #expect(parsed.brackets.count == 4)
      #expect(parsed.lines.last?.output.quote == 0)
    }
    #expect(ArtifactSourceSyntax.normalize("c++") == "cpp")
    #expect(ArtifactEditing.comment(for: "json") == nil)
    #expect(ArtifactEditing.comment(for: "lua") == .line("--"))
    let previous = try ArtifactSourceSyntax.parse("const oldIdentifier = 1;\n", language: "js")
    let changed = try ArtifactSourceSyntax.parse(
      "const newIdentifier = 1;\n", language: "js", previous: previous)
    #expect(!changed.words.contains("oldIdentifier") && changed.words.contains("newIdentifier"))
  }
}
