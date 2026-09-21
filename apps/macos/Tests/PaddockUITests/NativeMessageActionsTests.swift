import AppKit
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native message actions", .serialized) @MainActor
struct NativeMessageActionsTests {
  private func transcript() throws -> StudioState.NativeTranscript {
    try JSONDecoder().decode(
      StudioState.NativeTranscript.self,
      from: Data(
        #"""
        {"available":true,"notice":"","conversationId":"c","leafId":"answer","messages":[
        {"id":"question","role":"user","text":"Original question","reasoning":"","model":"","streaming":false,"stopped":false,"error":"","incomplete":false,"actions":{"edit":true,"retry":false,"continueReply":false,"branch":{"index":2,"count":3,"previous":"old","next":"new"}}},
        {"id":"answer","role":"assistant","text":"Partial answer","reasoning":"","model":"fixture","streaming":false,"stopped":false,"error":"","incomplete":true,"actions":{"edit":false,"retry":true,"continueReply":true,"branch":{"index":2,"count":3,"previous":"old-answer","next":"new-answer"}}}]}
        """#.utf8))
  }
  @Test func targetsContainOnlyDisplayedPathIdentifiers() throws {
    let projection = try transcript()
    let target = try #require(StudioMessageTarget(transcript: projection, messageId: "question"))
    #expect(
      target.payload(action: "retry") == [
        "action": .string("retry"), "conversationId": .string("c"), "leafId": .string("answer"),
        "messageId": .string("question"),
      ])
    #expect(StudioMessageTarget(transcript: projection, messageId: "off-path") == nil)
    let edit = StudioMessageEdit(target: target, text: "Original question")
    #expect(edit.canSubmit)
    var changed = edit
    changed.text = "  \n"
    #expect(!changed.canSubmit)
    changed.text = String(repeating: "🦄", count: 40000)
    #expect(!changed.canSubmit)
    #expect(changed.originalText == "Original question")
  }
  @Test func staleSubmissionAndReloadKeepAppOwnedEdit() async throws {
    let chat = StudioWorkspace(client: MessageFixture())
    let target = try #require(StudioMessageTarget(transcript: transcript(), messageId: "question"))
    chat.messageEdit = StudioMessageEdit(target: target, text: "Keep this edit")
    #expect(await !chat.submitMessageEdit())
    chat.reload()
    #expect(chat.messageEdit?.text == "Keep this edit")
    #expect(chat.error?.contains("message edits") == true)
    chat.cancelMessageEdit()
    #expect(!chat.hasMessageEdit)
  }
  @Test func fullActionFootersAndEditorFitBothAppearances() async throws {
    _ = NSApplication.shared
    let projection = try transcript()
    let chat = StudioWorkspace(client: MessageFixture())
    for width: CGFloat in [280, 760] {
      for dark in [false, true] {
        for message in projection.messages {
          let host = NSHostingController(
            rootView: NativeMessageFooter(
              message: message, workspace: chat,
              target: StudioMessageTarget(transcript: projection, messageId: message.id)
            )
            .preferredColorScheme(dark ? .dark : .light))
          let fitted = host.sizeThatFits(in: CGSize(width: width, height: 1000))
          #expect(fitted.width <= width + 1)
        }
        let target = try #require(
          StudioMessageTarget(transcript: projection, messageId: "question"))
        let edit = StudioMessageEdit(
          target: target, text: "Multiline question\nwith a small correction")
        chat.messageEdit = edit
        let host = NSHostingController(
          rootView: NativeMessageEditor(chat: chat, edit: edit)
            .preferredColorScheme(dark ? .dark : .light))
        let fitted = host.sizeThatFits(in: CGSize(width: width, height: 1000))
        #expect(fitted.width <= width + 1)
        #expect(fitted.height >= 150)
      }
    }
  }
  @Test func editorUsesTextKit2CaretUndoAndIMESafeCancel() async throws {
    _ = NSApplication.shared
    var text = "Original question"
    var sent = 0
    var cancelled = 0
    let root = StudioDraftEditor(
      text: Binding(get: { text }, set: { text = $0 }),
      onSend: { sent += 1 }, onCancel: { cancelled += 1 }, focusOnMount: true,
      editorID: "native-edit-text", editorLabel: "Edit message")
    let host = NSHostingView(rootView: root)
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 500, height: 300), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    window.makeKeyAndOrderFront(nil)
    defer { window.close() }
    func find(_ view: NSView) -> DraftTextView? {
      if let editor = view as? DraftTextView { return editor }
      return view.subviews.lazy.compactMap { find($0) }.first
    }
    for _ in 0..<100 {
      host.layoutSubtreeIfNeeded()
      if window.firstResponder is DraftTextView { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    let editor = try #require(find(host))
    #expect(editor.textLayoutManager != nil)
    #expect(editor.selectedRange() == NSRange(location: 17, length: 0))
    editor.insertText(" changed", replacementRange: editor.selectedRange())
    #expect(text == "Original question changed")
    editor.breakUndoCoalescing()
    try await Task.sleep(for: .milliseconds(50))
    #expect(editor.undoManager?.canUndo == true)
    editor.undoManager?.undo()
    try await Task.sleep(for: .milliseconds(50))
    #expect(editor.string == "Original question")
    #expect(text == "Original question")
    func key(_ code: UInt16, _ chars: String, _ flags: NSEvent.ModifierFlags = []) throws -> NSEvent
    {
      try #require(
        NSEvent.keyEvent(
          with: .keyDown, location: .zero, modifierFlags: flags, timestamp: 0,
          windowNumber: window.windowNumber, context: nil, characters: chars,
          charactersIgnoringModifiers: chars, isARepeat: false, keyCode: code))
    }
    editor.keyDown(with: try key(36, "\r", .shift))
    #expect(sent == 0)
    editor.setMarkedText(
      "かな", selectedRange: NSRange(location: 2, length: 0),
      replacementRange: NSRange(location: NSNotFound, length: 0))
    editor.keyDown(with: try key(53, "\u{1b}"))
    #expect(cancelled == 0)
    editor.unmarkText()
    editor.keyDown(with: try key(36, "\r"))
    editor.keyDown(with: try key(53, "\u{1b}"))
    #expect(sent == 1)
    #expect(cancelled == 1)
  }
}

private struct MessageFixture: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.core("Unused fixture") }
}
