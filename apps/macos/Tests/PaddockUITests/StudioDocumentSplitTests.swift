import AppKit
import PaddockStudio
import SwiftUI
import Testing
import WebKit

@testable import PaddockUI

@Suite("Native document split", .serialized) @MainActor
struct StudioDocumentSplitTests {
  @Test func rightArtifactReopensWithContentAfterConversationChanges() async throws {
    _ = NSApplication.shared
    let state = ArtifactSplitFixtureState()
    let host = NSHostingController(rootView: ArtifactSplitFixture(state: state))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1078, height: 650),
      styleMask: [.titled, .resizable], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setContentSize(NSSize(width: 1078, height: 650))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for cycle in 0..<5 {
      state.conversation = "artifact-\(cycle)"
      state.open = true
      try await Task.sleep(for: .milliseconds(120))
      host.view.layoutSubtreeIfNeeded()
      let split = try #require(splits(in: host.view).first { $0.right })
      #expect(!split.documentItem.isCollapsed)
      let preview = try #require(find(ArtifactPreviewMarker.self, in: split.documentHost.view))
      #expect(!preview.isHiddenOrHasHiddenAncestor)
      #expect(preview.visibleRect.width > 300 && preview.visibleRect.height > 500)
      #expect(abs(split.documentHost.view.frame.width - 460) < 2)
      state.open = false
      try await Task.sleep(for: .milliseconds(120))
      #expect(split.documentItem.isCollapsed)
      state.conversation = "other"
      try await Task.sleep(for: .milliseconds(60))
      state.conversation = "artifact-\(cycle)"
      try await Task.sleep(for: .milliseconds(60))
    }
  }
  @Test func nestedSplitDragDoesNotResizeWindowOrSidebar() async throws {
    _ = NSApplication.shared
    let state = SplitFixtureState()
    let host = NSHostingController(rootView: SplitFixture(state: state))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1200, height: 720),
      styleMask: [.titled, .resizable], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setContentSize(NSSize(width: 1200, height: 720))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(150))
    let split = try #require(documentSplit(in: host.view)?.delegate as? StudioSplitController)
    let outer = try #require(find(NSSplitView.self, in: host.view))
    let frame = window.frame
    let sidebarWidth = try #require(outer.subviews.first?.frame.width)
    let initialEditor = try #require(find(DraftTextView.self, in: split.chatHost.view))
    let initialEditorScroll = try #require(initialEditor.enclosingScrollView)
    #expect(abs(initialEditor.frame.width - initialEditorScroll.contentView.bounds.width) < 1)
    state.open = true
    try await Task.sleep(for: .milliseconds(150))
    for position: CGFloat in [400, 450, 500, 550, 500, 450, 400] {
      split.splitView.setPosition(position, ofDividerAt: 0)
      split.view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(50))
      #expect(window.frame == frame)
      #expect(abs(outer.subviews[0].frame.width - sidebarWidth) < 1)
      let documentFrame = split.documentHost.view.convert(
        split.documentHost.view.bounds, to: split.splitView)
      let chatFrame = split.chatHost.view.convert(split.chatHost.view.bounds, to: split.splitView)
      #expect(abs(documentFrame.minX) < 1)
      #expect(abs(documentFrame.width - position) < 1)
      #expect(abs(chatFrame.minX - documentFrame.maxX - split.splitView.dividerThickness) < 1)
      #expect(abs(split.view.bounds.height - 720) < 1)
      #expect(
        abs(
          split.chatHost.view.frame.width + split.documentHost.view.frame.width
            + split.splitView.dividerThickness - split.splitView.bounds.width) < 1)
      #expect(abs(state.column.width - min(760, split.chatHost.view.frame.width - 56)) < 1)
      let editor = try #require(find(DraftTextView.self, in: split.chatHost.view))
      let editorScroll = try #require(editor.enclosingScrollView)
      #expect(abs(editor.frame.width - editorScroll.contentView.bounds.width) < 1)
    }
    state.open = false
    try await Task.sleep(for: .milliseconds(150))
    #expect(window.frame == frame)
    #expect(abs(outer.subviews[0].frame.width - sidebarWidth) < 1)
  }

  @Test func retainsEditorWebViewSelectionFocusAndReopenWidth() async throws {
    _ = NSApplication.shared
    let split = StudioSplitController()
    let web = WKWebView()
    let chat = AnyView(
      StudioDraftEditor(
        text: .constant("Keep my draft and selection"), onSend: {}, onFiles: { _ in }))
    let document = AnyView(TestWeb(view: web))
    split.update(chat: chat, document: document, open: false)
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1100, height: 720),
      styleMask: [.titled, .resizable], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = split
    window.setContentSize(NSSize(width: 1100, height: 720))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(100))
    split.view.layoutSubtreeIfNeeded()
    let editor = try #require(find(DraftTextView.self, in: split.chatHost.view))
    let range = NSRange(location: 5, length: 8)
    editor.setSelectedRange(range)
    window.makeFirstResponder(editor)
    for _ in 0..<5 {
      split.update(chat: chat, document: document, open: true)
      try await Task.sleep(for: .milliseconds(80))
      split.view.layoutSubtreeIfNeeded()
      #expect(!split.documentItem.isCollapsed)
      #expect(find(DraftTextView.self, in: split.chatHost.view) === editor)
      #expect(find(WKWebView.self, in: split.documentHost.view) === web)
      #expect(editor.selectedRange() == range)
      split.splitView.setPosition(400, ofDividerAt: 0)
      split.view.layoutSubtreeIfNeeded()
      let chosen = split.documentHost.view.frame.width
      window.makeFirstResponder(web)
      split.update(chat: chat, document: document, open: false)
      try await Task.sleep(for: .milliseconds(60))
      #expect(split.documentItem.isCollapsed)
      #expect(window.firstResponder === editor)
      #expect(editor.selectedRange() == range)
      split.update(chat: chat, document: document, open: true)
      try await Task.sleep(for: .milliseconds(60))
      split.view.layoutSubtreeIfNeeded()
      #expect(abs(split.documentHost.view.frame.width - chosen) < 2)
      split.update(chat: chat, document: document, open: false)
    }
  }

  @Test func documentBadgeKeepsMetadataAndFitsNarrowColumn() throws {
    let document = try JSONDecoder().decode(
      StudioState.Document.self,
      from: Data(
        #"{"id":"second","name":"A long original document filename.pdf","kind":"pdf","pages":6,"pageRange":"2-4","textOnly":true}"#
          .utf8))
    #expect(document.id == "second")
    #expect(document.pageRange == "2-4")
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: NativeDocumentBadge(document: document, onOpen: {})
          .preferredColorScheme(dark ? .dark : .light))
      #expect(host.sizeThatFits(in: CGSize(width: 264, height: 200)).width <= 264)
    }
  }

  @Test func collapsedWorkspaceStillDeliversIncrementalUpdates() async throws {
    _ = NSApplication.shared
    let web = WKWebView()
    let split = StudioSplitController()
    split.update(
      chat: AnyView(Text("Native chat")), document: AnyView(TestWeb(view: web)), open: false)
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1100, height: 700), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = split
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    web.loadHTMLString("<html><body>Private timing fixture</body></html>", baseURL: nil)
    for _ in 0..<100 {
      if !web.isLoading { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    let elapsed =
      try await web.callAsyncJavaScript(
        "const start=performance.now();for(let i=0;i<10;i++)await new Promise(r=>setTimeout(r,32));return performance.now()-start",
        arguments: [:], in: nil, contentWorld: .page) as? Double
    #expect(split.documentItem.isCollapsed)
    #expect(
      try #require(elapsed) < 1000,
      "Hidden workspace must not acquire one-second-per-update background throttling")
  }

  @Test func documentResizeRetainsTranscriptSelectionAndReadingPosition() async throws {
    _ = NSApplication.shared
    let text = (1...80).map {
      "Paragraph \($0). Keep this reading position while the document pane opens and the lines wrap to the new column width."
    }.joined(separator: "\n\n")
    let bytes = try JSONSerialization.data(withJSONObject: [
      "available": true, "notice": "",
      "messages": [
        [
          "id": "reply", "role": "assistant", "text": text, "reasoning": "", "model": "fixture",
          "streaming": false, "stopped": false, "error": "", "incomplete": false,
        ]
      ],
    ])
    let transcript = try JSONDecoder().decode(StudioState.NativeTranscript.self, from: bytes)
    let chat = AnyView(
      GeometryReader { geometry in
        NativeStudioTranscript(
          transcript: transcript, columnWidth: max(0, geometry.size.width - 56), composerHeight: 0)
      })
    let document = AnyView(Text("Document fixture"))
    let split = StudioSplitController()
    split.update(chat: chat, document: document, open: false)
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1100, height: 700), styleMask: [.titled, .resizable],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = split
    window.setContentSize(NSSize(width: 1100, height: 700))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    var surface: NSTextView?
    for _ in 0..<100 {
      split.view.layoutSubtreeIfNeeded()
      surface = find(NSTextView.self, in: split.chatHost.view)
      if surface?.string.contains("Paragraph 80") == true { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    let original = try #require(surface)
    let scroll = try #require(find(NSScrollView.self, in: split.chatHost.view))
    #expect(
      scroll.contentInsets.bottom == 0, "An absent composer must not reserve a full-height bar")
    // TextKit's string arrives before SwiftUI publishes its measured height
    // and applies the initial tail anchor. Start the reading-position test
    // after that layout, not while its programmatic scroll is being replaced.
    try await Task.sleep(for: .milliseconds(200))
    split.view.layoutSubtreeIfNeeded()
    let selection = NSRange(location: 20, length: 24)
    original.setSelectedRange(selection)
    scroll.contentView.scroll(to: NSPoint(x: 0, y: 400))
    scroll.reflectScrolledClipView(scroll.contentView)
    try await Task.sleep(for: .milliseconds(100))
    let offset = scroll.documentVisibleRect.minY
    let originalWidth = split.chatHost.view.frame.width
    #expect(abs(offset - 400) < 2, "The fixture must be reading history, not pinned at the tail")
    split.update(chat: chat, document: document, open: true)
    try await Task.sleep(for: .milliseconds(150))
    split.view.layoutSubtreeIfNeeded()
    #expect(find(NSTextView.self, in: split.chatHost.view) === original)
    #expect(original.selectedRange() == selection)
    for position: CGFloat in [400, 520, 600, 460] {
      split.splitView.setPosition(position, ofDividerAt: 0)
      try await Task.sleep(for: .milliseconds(60))
      split.view.layoutSubtreeIfNeeded()
      #expect(find(NSTextView.self, in: split.chatHost.view) === original)
      #expect(original.selectedRange() == selection)
    }
    split.update(chat: chat, document: document, open: false)
    try await Task.sleep(for: .milliseconds(150))
    split.view.layoutSubtreeIfNeeded()
    #expect(find(NSScrollView.self, in: split.chatHost.view) === scroll)
    #expect(abs(split.chatHost.view.frame.width - originalWidth) < 2)
    #expect(abs(scroll.documentVisibleRect.minY - offset) < 2)
    #expect(original.selectedRange() == selection)
  }

  private func find<T: NSView>(_ type: T.Type, in view: NSView) -> T? {
    if let match = view as? T { return match }
    return view.subviews.lazy.compactMap { find(type, in: $0) }.first
  }
  private func documentSplit(in view: NSView) -> NSSplitView? {
    if let split = view as? NSSplitView,
      split.accessibilityIdentifier() == "studio-document-split"
    {
      return split
    }
    return view.subviews.lazy.compactMap { documentSplit(in: $0) }.first
  }
  private func splits(in view: NSView) -> [StudioSplitController] {
    let current = (view as? NSSplitView)?.delegate as? StudioSplitController
    return (current.map { [$0] } ?? []) + view.subviews.flatMap { splits(in: $0) }
  }
  @MainActor @Observable final class ArtifactSplitFixtureState {
    var open = false
    var conversation = "first"
  }
  private struct ArtifactSplitFixture: View {
    let state: ArtifactSplitFixtureState
    var body: some View {
      StudioDocumentSplit(open: false) {
        StudioDocumentSplit(open: state.open, right: true) {
          HStack(spacing: 0) {
            GeometryReader { geometry in
              VStack {
                Text("Native transcript \(state.conversation)")
                Spacer()
                Text("Composer")
              }
              .frame(width: geometry.size.width, height: geometry.size.height)
            }.id(state.conversation)
            if !state.open { Text("Open").frame(width: 40) }
          }
        } document: {
          if state.open { ArtifactPreviewMarkerView() }
        }
      } document: {
        EmptyView()
      }
    }
  }
  private final class ArtifactPreviewMarker: NSView {}
  private struct ArtifactPreviewMarkerView: NSViewRepresentable {
    func makeNSView(context: Context) -> ArtifactPreviewMarker { ArtifactPreviewMarker() }
    func updateNSView(_ view: ArtifactPreviewMarker, context: Context) {}
  }
  @MainActor @Observable final class SplitFixtureState {
    var open = false
    var column = CGRect.zero
    let web = WKWebView()
  }
  private struct SplitFixture: View {
    let state: SplitFixtureState
    var body: some View {
      GeometryReader { _ in
        HSplitView {
          Text("History").frame(minWidth: 200, idealWidth: 220, maxWidth: 260, maxHeight: .infinity)
          StudioDocumentSplit(open: state.open) {
            GeometryReader { geometry in
              let column = StudioColumnLayout.resolve(available: geometry.size.width, viewport: nil)
              VStack {
                Text("Markdown and composer column")
                Spacer()
                StudioDraftEditor(text: .constant(""), onSend: {}, onFiles: { _ in })
                  .frame(height: 72)
              }.frame(width: column.width)
                .offset(x: column.minX)
                .onGeometryChange(for: CGRect.self) { _ in
                  column
                } action: {
                  state.column = $0
                }
            }
          } document: {
            VStack(spacing: 0) {
              StudioDocumentHeader(document: nil, onClose: { state.open = false })
              TestWeb(view: state.web)
            }
          }
          .frame(minWidth: 640, maxWidth: .infinity, maxHeight: .infinity)
        }
      }.frame(minWidth: 900, minHeight: 650)
    }
  }
  private struct TestWeb: NSViewRepresentable {
    let view: WKWebView
    func makeNSView(context: Context) -> WKWebView { view }
    func updateNSView(_ view: WKWebView, context: Context) {}
  }
}
