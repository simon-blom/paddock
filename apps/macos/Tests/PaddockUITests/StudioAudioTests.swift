import AppKit
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native speech composer", .serialized) @MainActor
struct StudioAudioTests {
  @Test func nativeAudioNavigationNeverCreatesAHiddenWebWorkspace() async throws {
    _ = NSApplication.shared
    let client = AudioNoCore()
    let chat = StudioWorkspace(client: client)
    let model = WorkspaceModel(client: client, preparedStudio: chat)
    let host = NSHostingController(rootView: WorkspaceView(model: model))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1000, height: 800),
      styleMask: [.titled], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setContentSize(NSSize(width: 1000, height: 800))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    defer { window.close() }
    for location in ["settings", "manager", "chat", "settings", "manager"] {
      var nav = WorkspaceNavigation()
      if location == "manager" {
        nav.showManager(.downloads)
      } else {
        nav.mode = .studio
        nav.studio = location == "settings" ? .settings : .newChat
      }
      host.rootView = WorkspaceView(model: model, navigation: nav)
      try await Task.sleep(for: .milliseconds(100))
      host.view.layoutSubtreeIfNeeded()
      // Audio is AVAudioEngine + native transport, not a hidden WKWebView.
      // Accessing the retired webView getter here would itself instantiate the
      // viewer we are asserting is absent. Only document viewers may create it.
      #expect(!chat.hasWebViewer, "Navigation created WebKit on \(location)")
      #expect(model.chat === chat, "Navigation replaced the native audio owner")
    }
    await chat.shutdown()
  }

  @Test func finalizedDictationPreservesCaretUndoAndTextKitTwo() async throws {
    _ = NSApplication.shared
    var draft = "Correct this sentence."
    let host = NSHostingController(
      rootView: StudioDraftEditor(
        text: Binding(get: { draft }, set: { draft = $0 }), onSend: {}
      ).frame(width: 320, height: 72))
    host.view.frame = NSRect(x: 0, y: 0, width: 320, height: 72)
    host.view.layoutSubtreeIfNeeded()
    func find(_ view: NSView) -> DraftTextView? {
      (view as? DraftTextView) ?? view.subviews.lazy.compactMap(find).first
    }
    let editor = try #require(find(host.view))
    editor.setSelectedRange(NSRange(location: 2, length: 4))
    #expect(editor.appendDictated("  Hej världen.  "))
    #expect(draft == "Correct this sentence. Hej världen.")
    #expect(editor.selectedRange() == NSRange(location: 2, length: 4))
    #expect(editor.textLayoutManager != nil)
    #expect(editor.undoManager?.canUndo == true)
    editor.undoManager?.undo()
    #expect(editor.string == "Correct this sentence.")
    editor.setMarkedText(
      "編集中", selectedRange: NSRange(location: 0, length: 0),
      replacementRange: NSRange(location: 0, length: 0))
    #expect(!editor.appendDictated("Must wait for IME"))
    editor.unmarkText()
    editor.isEditable = false
    #expect(!editor.appendDictated("Must wait for editable draft"))
  }

  @Test func microphoneModesHaveDistinctActionsAndBoundedClocks() {
    #expect(StudioMicrophoneSettings.label("live") == "Live")
    #expect(StudioMicrophoneSettings.label("record") == "Record and send")
    #expect(StudioMicrophoneSettings.label("dictate") == "Transcribe into the composer")
    #expect(StudioMicrophoneStatus.clock(61.8) == "1:01")
    #expect(StudioMicrophoneStatus.clock(.nan) == "0:00")
    #expect(StudioMicrophoneStatus.clock(-1) == "0:00")
  }
}

private struct AudioNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
}
