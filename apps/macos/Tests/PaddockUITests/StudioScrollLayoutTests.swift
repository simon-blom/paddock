import AppKit
import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Full-height native transcript", .serialized) @MainActor
struct StudioScrollLayoutTests {
  @Test func remountingMeasuredRowsDoesNotRequestAnotherScroll() {
    let intent = TranscriptScrollIntent()
    let size = CGSize(width: 600, height: 1200)
    intent.didLayout("answer", size: size)
    #expect(intent.layoutRevision == 1)
    for _ in 0..<100 { intent.didLayout("answer", size: size) }
    #expect(intent.layoutRevision == 1)
    intent.didLayout("answer", size: CGSize(width: 600, height: 1230))
    #expect(intent.layoutRevision == 2)
  }

  @Test func actualLayoutGrowthCannotPullAReaderAwayFromAnOpenDisclosure() {
    let intent = TranscriptScrollIntent()
    intent.didLayout("answer", size: CGSize(width: 600, height: 1200))
    intent.reveal()
    intent.didLayout("answer", size: CGSize(width: 600, height: 1500))
    #expect(!intent.pinned && intent.readingDisclosure)
    #expect(intent.layoutRevision == 1)
    intent.pinned = true
    intent.didLayout("answer", size: CGSize(width: 600, height: 1500))
    #expect(intent.layoutRevision == 1, "Measurements still update while the reader is unpinned")
  }

  @Test func unavailableRowMeasurementsDoNotStartAFollowLoop() {
    let intent = TranscriptScrollIntent()
    for size in [
      CGSize.zero, CGSize(width: CGFloat.nan, height: 100),
      CGSize(width: 600, height: CGFloat.infinity), CGSize(width: 600, height: -1),
    ] {
      intent.didLayout("answer", size: size)
    }
    #expect(intent.layoutRevision == 0)
  }

  @Test func latestButtonClearsComposerAndStaysInsideItsColumn() throws {
    _ = NSApplication.shared
    for (width, column): (CGFloat, CGFloat) in [(820, 760), (1280, 1224), (420, 364)] {
      for composer: CGFloat in [120, 172, 320, 420] {
        let viewport: CGFloat = 700
        let renderer = ImageRenderer(
          content:
            NativeTranscriptLatestButton(columnWidth: column, composerHeight: composer, action: {})
            .frame(width: width, height: viewport, alignment: .bottom))
        renderer.scale = 1
        let image = try #require(renderer.nsImage)
        let data = try #require(image.tiffRepresentation)
        let pixels = try #require(NSBitmapImageRep(data: data))
        var bottom = -1
        var right = -1
        var top = pixels.pixelsHigh
        for y in 0..<pixels.pixelsHigh {
          for x in 0..<pixels.pixelsWide {
            if (pixels.colorAt(x: x, y: y)?.alphaComponent ?? 0) > 0.2 {
              top = min(top, y)
              bottom = max(bottom, y)
              right = max(right, x)
            }
          }
        }
        #expect(top >= 0 && bottom - top >= 24, "The complete button must be visible")
        #expect(
          abs(CGFloat(bottom + 1) - (viewport - composer - 12)) <= 1,
          "Rendered button must track composer growth with a 12-point gap")
        #expect(
          abs(CGFloat(right + 1) - ((width + column) / 2 - 12)) <= 1,
          "Align inside the composer column, including wide Compare and narrow windows")
      }
    }
  }

  @Test func fullViewportClearsGrowingComposerAndFollowsOnlyWhenPinned() async throws {
    _ = NSApplication.shared
    let state = TranscriptLayoutFixture()
    let host = NSHostingController(rootView: TranscriptLayoutView(state: state))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 820, height: 700),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    host.sizingOptions = []
    window.contentViewController = host
    window.setContentSize(NSSize(width: 820, height: 700))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)  // Offscreen; enables SwiftUI's scroll-geometry callbacks.
    defer { window.close() }
    try await settle(host.view)
    let scroll = try #require(find(NSScrollView.self, in: host.view))
    let document = try #require(scroll.documentView)
    // Native Markdown parses asynchronously. Wait for this fixture's real
    // text layout, not its one-line placeholder, before testing tail geometry.
    for _ in 0..<20 {
      if document.bounds.height > 2000 { break }
      try await settle(host.view)
    }
    #expect(document.bounds.height > 2000)
    #expect(abs(scroll.contentView.bounds.height - 700) < 1)
    #expect(
      scroll.scrollerInsets.top == StudioConversationSpacing.scrollIndicatorInset
        && scroll.scrollerInsets.bottom == StudioConversationSpacing.scrollIndicatorInset,
      "Only corner clearance, not titlebar/composer height, should inset the indicator")

    for height: CGFloat in [172, 320, 120] {
      state.composer = height
      try await settle(host.view)
      #expect(scroll.documentView === document)
      #expect(abs(scroll.contentView.bounds.height - 700) < 1)
      #expect(
        abs(document.bounds.maxY + scroll.contentInsets.bottom - scroll.documentVisibleRect.maxY)
          < 2,
        "Tail must stay pinned when the composer grows or shrinks")
      let text = try #require(find(NSTextView.self, in: document))
      let textRect = text.convert(text.bounds, to: document)
      #expect(
        scroll.documentVisibleRect.maxY - textRect.maxY >= height
          + StudioConversationSpacing.composerGap,
        "The final text and message actions must stop above the floating composer")
    }

    state.paragraphs += 4
    try await settle(host.view)
    #expect(
      abs(document.bounds.maxY + scroll.contentInsets.bottom - scroll.documentVisibleRect.maxY) < 2)
    for height: CGFloat in [520, 820, 700] {
      window.setContentSize(NSSize(width: 820, height: height))
      try await settle(host.view)
      #expect(abs(scroll.contentView.bounds.height - height) < 1)
      #expect(
        abs(document.bounds.maxY + scroll.contentInsets.bottom - scroll.documentVisibleRect.maxY)
          < 2)
    }
    // A no-wheel-phase scroll, as used by accessibility and keyboard jumps.
    scroll.contentView.scroll(to: NSPoint(x: 0, y: 300))
    scroll.reflectScrolledClipView(scroll.contentView)
    try await settle(host.view)
    try await settle(host.view)
    let text = try #require(find(NSTextView.self, in: document))
    let selection = NSRange(location: 12, length: 20)
    text.setSelectedRange(selection)
    let offset = scroll.documentVisibleRect.minY
    #expect(
      document.bounds.maxY + scroll.contentInsets.bottom - scroll.documentVisibleRect.maxY > 100)
    state.composer = 300
    state.paragraphs += 3
    try await settle(host.view)
    #expect(
      abs(scroll.documentVisibleRect.minY - offset) < 2,
      "Reading earlier text must not jump to the bottom on streaming or draft growth")
    #expect(find(NSTextView.self, in: document) === text)
    #expect(text.selectedRange() == selection)
  }

  @Test func shortConversationStartsAtTopWithoutASeparateComposerViewport() async throws {
    let state = TranscriptLayoutFixture()
    state.paragraphs = 1
    let host = NSHostingController(rootView: TranscriptLayoutView(state: state))
    host.view.frame = NSRect(x: 0, y: 0, width: 820, height: 700)
    try await settle(host.view)
    let scroll = try #require(find(NSScrollView.self, in: host.view))
    #expect(abs(scroll.contentView.bounds.height - 700) < 1)
    #expect(abs(scroll.documentVisibleRect.minY) < 1)
  }

  @Test func readingAnchorAccountsForContentBehindNativeBars() async throws {
    let state = TranscriptLayoutFixture()
    state.composer = 200
    let host = NSHostingController(rootView: TranscriptLayoutView(state: state))
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 820, height: 700),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setContentSize(NSSize(width: 820, height: 700))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await settle(host.view)
    let scroll = try #require(find(NSScrollView.self, in: host.view))
    let document = try #require(scroll.documentView)
    for _ in 0..<20 {
      if document.bounds.height > 2000 { break }
      try await settle(host.view)
    }
    // Only 80 points from the usable reading area's bottom. The old test
    // incorrectly treated this as pinned because 200 points are behind a bar.
    let maximum = document.bounds.maxY + scroll.contentInsets.bottom - scroll.bounds.height
    scroll.contentView.scroll(to: NSPoint(x: 0, y: maximum - 80))
    scroll.reflectScrolledClipView(scroll.contentView)
    try await settle(host.view)
    let anchor = try #require(StudioReadingAnchor.capture(in: host.view))
    state.composer = 280
    window.setContentSize(NSSize(width: 700, height: 700))
    try await settle(host.view)
    anchor.restore()
    try await settle(host.view)
    let restored = try #require(StudioReadingAnchor.capture(in: host.view))
    #expect(restored.text === anchor.text)
    #expect(abs(restored.lineOffset - anchor.lineOffset) < 2)
  }

  @Test func nativeBarsOwnTitlebarAndComposerClearanceInBothAppearances() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      let state = TranscriptLayoutFixture()
      let host = NSHostingController(rootView: TranscriptLayoutView(state: state))
      host.sizingOptions = []
      host.safeAreaRegions = []
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 900, height: 700),
        styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
        backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.titleVisibility = .hidden
      window.titlebarAppearsTransparent = true
      window.toolbar = NSToolbar(identifier: "TranscriptChromeTest")
      window.toolbarStyle = .unifiedCompact
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      window.contentViewController = host
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for (width, leading, trailing, comparison): (CGFloat, CGFloat, CGFloat, Bool) in [
        (900, 0, 0, false), (900, 240, 0, false), (1440, 240, 420, false),
        (1280, 0, 0, true), (900, 0, 0, true),
      ] {
        state.leadingWidth = leading
        state.trailingWidth = trailing
        state.comparison = comparison
        window.setContentSize(NSSize(width: width, height: 700))
        try await settle(host.view)
        let scroll = try #require(find(NSScrollView.self, in: host.view))
        let viewport = scroll.convert(scroll.bounds, to: nil)
        #expect(abs(viewport.maxY - window.contentLayoutRect.maxY) < 1)
        #expect(scroll.contentInsets.top == 0)
        #expect(abs(scroll.contentInsets.bottom - state.composer) < 1)
        #expect(abs(viewport.minY - host.view.convert(host.view.bounds, to: nil).minY) < 1)
        let probe = try #require(find(StudioWindowChromeProbeView.self, in: host.view))
        #expect(probe.bounds.height == 0, "Measuring chrome must not paint over the transcript")
        #expect(
          scroll.scrollerInsets.top == StudioConversationSpacing.scrollIndicatorInset
            && scroll.scrollerInsets.bottom == StudioConversationSpacing.scrollIndicatorInset)
        scroll.contentView.scroll(to: NSPoint(x: 0, y: 200))
        scroll.reflectScrolledClipView(scroll.contentView)
        try await settle(host.view)
        let text = try #require(find(NSTextView.self, in: scroll))
        let selected = NSRange(location: 12, length: 20)
        text.setSelectedRange(selected)
        let readingOffset = scroll.documentVisibleRect.minY + scroll.contentInsets.top
        window.toolbar?.isVisible = false
        try await settle(host.view)
        #expect(
          abs(scroll.convert(scroll.bounds, to: nil).maxY - window.contentLayoutRect.maxY) < 1)
        #expect(scroll.contentInsets.top == 0)
        #expect(text.selectedRange() == selected)
        #expect(
          abs(scroll.documentVisibleRect.minY + scroll.contentInsets.top - readingOffset) < 2,
          "System inset changes must preserve the first unobscured content position")
        window.toolbar?.isVisible = true
        try await settle(host.view)
        scroll.contentView.scroll(to: NSPoint(x: 0, y: -scroll.contentInsets.top))
        scroll.reflectScrolledClipView(scroll.contentView)
        try await settle(host.view)
        let firstText = try #require(find(NSTextView.self, in: scroll))
        #expect(
          firstText.convert(firstText.bounds, to: nil).maxY
            <= window.contentLayoutRect.maxY - StudioConversationSpacing.edgeInset + 1)
        #expect(find(StudioWindowChromeProbeView.self, in: host.view) === probe)
      }
    }
  }

  @Test func scrollingTablePastTitlebarDoesNotRelayoutTheMessage() async throws {
    _ = NSApplication.shared
    let state = TranscriptLayoutFixture()
    state.leadingWidth = 260
    state.markdown = """
      # Model comparison

      | Feature | Laptop model | Larger model |
      | --- | --- | --- |
      \((1...14).map { "| Feature \($0) | **Local inference** with a longer description that wraps across lines | Another long description with `code` and details |" }.joined(separator: "\n"))

      \((1...12).map { "## Section \($0)\n\nA paragraph after the table. " + String(repeating: "Native text stays readable while scrolling. ", count: 8) }.joined(separator: "\n\n"))
      """
    let host = NSHostingController(rootView: TranscriptLayoutView(state: state))
    host.sizingOptions = []
    host.safeAreaRegions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 902, height: 741),
      styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.titleVisibility = .hidden
    window.titlebarAppearsTransparent = true
    window.toolbar = NSToolbar(identifier: "ScrollingTableTest")
    window.toolbarStyle = .unifiedCompact
    window.contentViewController = host
    window.setContentSize(NSSize(width: 902, height: 741))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await settle(host.view)
    let scroll = try #require(find(NSScrollView.self, in: host.view))
    let document = try #require(scroll.documentView)
    for _ in 0..<20 {
      if document.bounds.height > 2500 { break }
      try await settle(host.view)
    }
    #expect(document.bounds.height > 2500)
    for sidebar: CGFloat in [260, 0] {
      state.leadingWidth = sidebar
      try await settle(host.view)
      let initialHeight = document.bounds.height
      let text = try #require(find(NSTextView.self, in: document))
      let selection = NSRange(location: 0, length: 10)
      text.setSelectedRange(selection)
      var minimumOffset = scroll.documentVisibleRect.minY
      var maximumOffset = minimumOffset
      for step in 0..<40 {
        let point = scroll.convert(NSPoint(x: scroll.bounds.midX, y: 180), to: nil)
        let hover = try #require(
          NSEvent.mouseEvent(
            with: .mouseMoved, location: point,
            modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 0,
            pressure: 0))
        window.sendEvent(hover)
        let event = try #require(
          CGEvent(
            scrollWheelEvent2Source: nil, units: .pixel,
            wheelCount: 1, wheel1: step < 20 ? 180 : -180, wheel2: 0, wheel3: 0))
        // Exercise the view under the pointer, not the scroll view directly:
        // nested Markdown hosting views must forward wheel input correctly.
        // This stays inside the fixture; no global HID events are posted.
        let hitPoint = host.view.superview?.convert(point, from: nil) ?? point
        let target = try #require(host.view.hitTest(hitPoint))
        target.scrollWheel(with: try #require(NSEvent(cgEvent: event)))
        try await settle(host.view)
        minimumOffset = min(minimumOffset, scroll.documentVisibleRect.minY)
        maximumOffset = max(maximumOffset, scroll.documentVisibleRect.minY)
        #expect(
          abs(document.bounds.height - initialHeight) < 1,
          "Scrolling must not change the Markdown's measured height")
        #expect(text.selectedRange() == selection)
      }
      #expect(
        maximumOffset - minimumOffset > 1000, "Exercise real scrolling, not just event delivery")
    }
  }

  private func settle(_ view: NSView) async throws {
    for _ in 0..<6 {
      view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(25))
    }
  }
  private func find<T: NSView>(_ type: T.Type, in view: NSView) -> T? {
    (view as? T) ?? view.subviews.lazy.compactMap { find(type, in: $0) }.first
  }
}

@MainActor @Observable private final class TranscriptLayoutFixture {
  var composer: CGFloat = 172
  var paragraphs = 60
  var leadingWidth: CGFloat = 0
  var trailingWidth: CGFloat = 0
  var comparison = false
  var markdown: String?
  var transcript: StudioState.NativeTranscript {
    let text =
      markdown
      ?? (1...paragraphs).map {
        "Paragraph \($0). Reading position and selection stay intact while the native composer grows and the answer continues streaming."
      }.joined(separator: "\n\n")
    var message: [String: Any] = [
      "id": "reply", "role": "assistant", "text": text, "reasoning": "",
      "model": "fixture", "streaming": false, "stopped": false, "error": "",
      "incomplete": false,
    ]
    if comparison { message["group"] = "compare" }
    var messages = [message]
    if comparison {
      message["id"] = "reply-2"
      messages.append(message)
    }
    let bytes = try! JSONSerialization.data(withJSONObject: [
      "available": true, "notice": "",
      "messages": messages,
    ])
    return try! JSONDecoder().decode(StudioState.NativeTranscript.self, from: bytes)
  }
}

private struct TranscriptLayoutView: View {
  @Bindable var state: TranscriptLayoutFixture
  var body: some View {
    HStack(spacing: 0) {
      Color.gray.frame(width: state.leadingWidth)
      GeometryReader { geometry in
        let column = StudioColumnLayout.resolve(
          available: geometry.size.width, viewport: nil, comparison: state.comparison)
        StudioConversationChrome {
          NativeStudioTranscript(
            transcript: state.transcript, columnWidth: column.width, composerHeight: state.composer,
            composer: AnyView(
              Color.gray.frame(width: column.width, height: state.composer).allowsHitTesting(false))
          )
        }
      }
      Color.gray.frame(width: state.trailingWidth)
    }
  }
}
