import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Column-owned native window chrome", .serialized) @MainActor
struct WorkspaceColumnChromeTests {
  @Test func productionConversationUsesFullHeightWithEitherSidebarState() async throws {
    _ = NSApplication.shared
    let client = ChromeNoCore()
    let chat = StudioWorkspace(client: client)
    let model = WorkspaceModel(client: client, preparedStudio: chat)
    chat.apply(try artifactState())
    model.navigation.sidebarVisible = true
    let host = NSHostingController(rootView: WorkspaceView(model: model))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1800, height: 800),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbarStyle = .unifiedCompact
    window.toolbar = NSToolbar(identifier: "production-compare-chrome")
    window.contentViewController = host
    window.setContentSize(NSSize(width: 1800, height: 800))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(150))
    host.view.layoutSubtreeIfNeeded()
    let split = try #require(
      allViews(host.view).compactMap { ($0 as? NSSplitView)?.delegate as? StudioSplitController }
        .first { $0.right })
    for sidebar in [true, false, true] {
      model.navigation.sidebarVisible = sidebar
      for dark in [false, true] {
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        try await Task.sleep(for: .milliseconds(80))
        host.view.layoutSubtreeIfNeeded()
        split.splitView.setPosition(600, ofDividerAt: 0)
        try await Task.sleep(for: .milliseconds(100))
        host.view.layoutSubtreeIfNeeded()
        let text = try #require(
          allViews(split.chatHost.view).compactMap { $0 as? NSTextView }
            .first { $0.string.hasPrefix("Full-height conversation fixture") })
        let scroll = try #require(text.enclosingScrollView)
        let frame = scroll.convert(scroll.bounds, to: host.view)
        #expect(abs(frame.minY) < 1, "The transcript viewport must reach the top: \(frame)")
        #expect(
          abs(frame.height - host.view.bounds.height) < 1,
          "The composer must overlay, not shrink, the transcript")
        #expect(scroll.scrollerInsets.top == 0 && scroll.scrollerInsets.bottom == 0)
        let scroller = try #require(scroll.verticalScroller)
        let track = scroller.convert(scroller.bounds, to: host.view)
        #expect(abs(track.minY) <= 2, "The real indicator must reach the window top: \(track)")
        #expect(
          abs(track.maxY - host.view.bounds.height) <= 2,
          "The real indicator must reach the window bottom: \(track)")
      }
    }
    // Exercise production headers, not replacement close closures. Closing a
    // neighbour must leave the other preview open, then collapse only the last.
    let previews = try #require(
      allViews(split.documentHost.view).compactMap { $0 as? NSSplitView }
        .first { $0.arrangedSubviews.count == 2 })
    let first = try #require(previews.arrangedSubviews.first)
    try clickClose(in: first, window: window)
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    #expect(chat.presentedArtifacts.map(\.id) == ["artifact-1"])
    #expect(chat.selectedArtifactId == "artifact-1")
    #expect(!split.documentItem.isCollapsed)
    // The one remaining preview uses the regular single-pane layout.
    try clickClose(in: split.documentHost.view, window: window)
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    #expect(chat.selectedArtifactId == nil && chat.presentedArtifacts.isEmpty)
    #expect(split.documentItem.isCollapsed)
    let transcriptText = try #require(
      allViews(split.chatHost.view).compactMap { $0 as? NSTextView }
        .first { $0.string.hasPrefix("Full-height conversation fixture") })
    let transcriptScroll = try #require(transcriptText.enclosingScrollView)
    #expect(
      abs(transcriptScroll.bounds.width - split.chatHost.view.bounds.width) < 1,
      "Closing the last preview must not insert a new rail beside the conversation")
    await model.shutdown()
  }

  private func clickClose(in pane: NSView, window: NSWindow) throws {
    let point = pane.convert(
      NSPoint(
        x: pane.bounds.maxX - 23,
        y: pane.isFlipped ? pane.bounds.minY + 21 : pane.bounds.maxY - 21), to: nil)
    for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
      window.sendEvent(
        try #require(
          NSEvent.mouseEvent(
            with: type, location: point,
            modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: window.windowNumber, context: nil, eventNumber: 1,
            clickCount: 1, pressure: 1)))
    }
  }

  @Test func fullHeightPreviewHeaderReceivesRealMouseClicks() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: ChromeNoCore())
    for dark in [false, true] {
      let state = ChromeFixtureState()
      let host = NSHostingController(
        rootView: ChromeFixture(state: state, model: model)
          .environment(\.colorScheme, dark ? .dark : .light))
      host.sizingOptions = []
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 1200, height: 700),
        styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
        backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.titleVisibility = .hidden
      window.titlebarAppearsTransparent = true
      window.toolbarStyle = .unifiedCompact
      window.toolbar = NSToolbar(identifier: "column-hit-test")
      window.contentViewController = host
      window.setContentSize(NSSize(width: 1200, height: 700))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for sidebar in [true, false, true] {
        model.navigation.sidebarVisible = sidebar
        for width: CGFloat in [1200, 1000, 1200] {
          window.setContentSize(NSSize(width: width, height: 700))
          try await Task.sleep(for: .milliseconds(100))
          host.view.layoutSubtreeIfNeeded()
          let split = try #require(find(NSSplitView.self, in: host.view))
          let splitRect = split.convert(split.bounds, to: host.view)
          #expect(abs(splitRect.minY) < 1)
          #expect(abs(splitRect.height - host.view.bounds.height) < 1)
          let header = state.probe.convert(state.probe.bounds, to: host.view)
          #expect(abs(header.minY) < 1, "No empty title-bar row above the preview: \(header)")
          #expect(header.height >= 38 && header.height < 60)
          #expect(window.contentLayoutRect.height < host.view.bounds.height)
          // Last header button is the native Close action. Exercise window event
          // dispatch, not AXPress, which bypasses title-bar hit interception.
          let point = state.probe.convert(
            NSPoint(x: state.probe.bounds.maxX - 23, y: state.probe.bounds.midY), to: nil)
          let before = state.hideCount
          for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            window.sendEvent(
              try #require(
                NSEvent.mouseEvent(
                  with: type, location: point,
                  modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                  windowNumber: window.windowNumber, context: nil, eventNumber: 1,
                  clickCount: 1, pressure: 1)))
          }
          try await Task.sleep(for: .milliseconds(30))
          #expect(
            state.hideCount == before + 1, "Native title bar must not intercept preview controls")
        }
      }
      // Compare introduces HSplitView-owned hosting roots inside our retained
      // split. They must not reapply NSWindow's safe area to each writer pane.
      model.navigation.sidebarVisible = true
      window.setContentSize(NSSize(width: 1700, height: 700))
      state.comparison = true
      try await Task.sleep(for: .milliseconds(100))
      host.view.layoutSubtreeIfNeeded()
      let split = try #require(find(NSSplitView.self, in: host.view))
      split.setPosition(600, ofDividerAt: 0)
      try await Task.sleep(for: .milliseconds(100))
      host.view.layoutSubtreeIfNeeded()
      for probe in [state.probe, state.secondProbe] {
        let header = probe.convert(probe.bounds, to: host.view)
        #expect(abs(header.minY) < 1, "Compare header must reach the top too: \(header)")
        #expect(header.width >= 340)
        let before = state.hideCount
        let point = probe.convert(NSPoint(x: probe.bounds.maxX - 23, y: probe.bounds.midY), to: nil)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
          window.sendEvent(
            try #require(
              NSEvent.mouseEvent(
                with: type, location: point,
                modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                windowNumber: window.windowNumber, context: nil, eventNumber: 1,
                clickCount: 1, pressure: 1)))
        }
        try await Task.sleep(for: .milliseconds(30))
        #expect(state.hideCount == before + 1)
      }
      state.comparison = false
    }
    await model.shutdown()
  }

  private func find<T: NSView>(_ type: T.Type, in view: NSView) -> T? {
    (view as? T) ?? view.subviews.lazy.compactMap { find(type, in: $0) }.first
  }
  private func allViews(_ root: NSView) -> [NSView] { [root] + root.subviews.flatMap(allViews) }
  private func artifactState() throws -> StudioState {
    let artifacts: [[String: Any]] = (0..<2).map { i in
      [
        "id": "artifact-\(i)", "model": "writer-\(i)", "kind": "html", "title": "Page \(i)",
        "language": "html", "versions": 1, "updatedAt": 1,
      ]
    }
    let fields: [String: Any] = [
      "version": 1, "revision": 1, "history": [], "models": [], "selectedModels": [],
      "busy": false, "loading": false, "error": "", "previewing": true, "unsavedEdits": false,
      "viewport": ["left": 0, "width": 760], "settings": [:], "tools": [],
      "conversation": [
        "id": "chrome", "title": "Chrome fixture", "model": "local", "messageCount": 0,
      ],
      "capabilities": [
        "reasoning": "", "levels": [], "reasoningDefault": "", "reasoningOff": false,
        "preserveThinking": false, "thinkingBudget": false, "webSearch": false, "vision": false,
        "context": 4096, "ocrModes": [], "docParser": false, "pdfRaster": false,
      ],
      "nativeArtifacts": artifacts, "nativeArtifactsPaneOpen": true,
      "nativeTranscript": [
        "available": true, "notice": "",
        "messages": [
          [
            "id": "message", "role": "assistant", "model": "local",
            "text":
              "Full-height conversation fixture\n\n"
              + (1...30).map {
                "Paragraph \($0). The scrollbar track must span the window, while content clears the composer and window controls."
              }.joined(separator: "\n\n"),
            "reasoning": "", "streaming": false, "stopped": false, "error": "", "incomplete": false,
          ]
        ],
      ],
    ]
    return try JSONDecoder().decode(
      StudioState.self, from: JSONSerialization.data(withJSONObject: fields))
  }
}

@Observable @MainActor private final class ChromeFixtureState {
  var source = false
  var version = 0
  var hideCount = 0
  var comparison = false
  let probe = NSView()
  let secondProbe = NSView()
}

private struct ChromeFixture: View {
  @Bindable var state: ChromeFixtureState
  @Bindable var model: WorkspaceModel
  var body: some View {
    GeometryReader { geometry in
      HStack(spacing: 0) {
        if model.navigation.showsSidebar {
          VStack(alignment: .leading) {
            Text("Chats")
            Spacer()
          }.padding(.top, geometry.safeAreaInsets.top).frame(width: 260)
        }
        StudioDocumentSplit(open: true, right: true) {
          Color.clear.padding(.top, geometry.safeAreaInsets.top)
        } document: {
          if state.comparison {
            HSplitView {
              pane(state.probe)
              pane(state.secondProbe)
            }
          } else {
            pane(state.probe)
          }
        }
      }.ignoresSafeArea(.container, edges: .top)
    }
    .toolbar {
      WorkspaceToolbar(navigation: $model.navigation, model: model, onNewChat: {}, onStart: {})
    }
    .toolbarBackgroundVisibility(.hidden, for: .windowToolbar)
  }
  private func pane(_ probe: NSView) -> some View {
    VStack(spacing: 0) {
      NativeArtifactHeader(
        writer: ("", ""), model: "local", title: "Page",
        source: $state.source, version: $state.version, versions: [],
        dirty: false, saving: false, available: true, copied: false,
        save: {}, revert: {}, copy: {}, download: {}, close: { state.hideCount += 1 }
      )
      .background(ChromeFrameProbe(view: probe))
      Color.clear
    }.frame(minWidth: 340).fullHeightWorkspaceColumn()
  }
}

private struct ChromeFrameProbe: NSViewRepresentable {
  let view: NSView
  func makeNSView(context: Context) -> NSView { view }
  func updateNSView(_ nsView: NSView, context: Context) {}
}

private struct ChromeNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
}
