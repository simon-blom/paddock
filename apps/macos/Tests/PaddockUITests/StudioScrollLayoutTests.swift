import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Full-height native transcript", .serialized) @MainActor
struct StudioScrollLayoutTests {
  @Test func conversationEdgeInsetsMatchExceptWhereWindowControlsNeedClearance() {
    #expect(
      StudioConversationSpacing.topInset(windowControls: 0) == StudioConversationSpacing.edgeInset)
    #expect(StudioConversationSpacing.topInset(windowControls: 38) == 50)
    #expect(StudioConversationSpacing.topInset(windowControls: 0, hasGraphAction: true) == 40)
    #expect(StudioConversationSpacing.topInset(windowControls: 38, hasGraphAction: true) == 78)
  }

  @Test func bottomGapCannotExposeTheScrollingTranscript() throws {
    for dark in [false, true] {
      let renderer = ImageRenderer(
        content: ZStack(alignment: .bottom) {
          Color(red: 1, green: 0, blue: 0)
          StudioComposerBottomCover().frame(width: 80)
        }.frame(width: 120, height: 100).environment(\.colorScheme, dark ? .dark : .light))
      renderer.scale = 1
      let data = try #require(renderer.nsImage?.tiffRepresentation)
      let pixels = try #require(NSBitmapImageRep(data: data))
      let exposed = try #require(pixels.colorAt(x: 60, y: 79)?.usingColorSpace(.deviceRGB))
      let covered = try #require(pixels.colorAt(x: 60, y: 90)?.usingColorSpace(.deviceRGB))
      let gutter = try #require(pixels.colorAt(x: 115, y: 90)?.usingColorSpace(.deviceRGB))
      // Device-RGB conversion can lift green/blue on a wide-gamut display.
      // Require the uncovered fixture's saturated red, not exact channel bytes.
      #expect(exposed.redComponent > exposed.greenComponent + 0.5)
      #expect(exposed.redComponent > exposed.blueComponent + 0.5)
      #expect(abs(covered.redComponent - covered.greenComponent) < 0.03)
      #expect(abs(covered.greenComponent - covered.blueComponent) < 0.03)
      #expect(covered.alphaComponent > 0.99)
      #expect(dark ? covered.redComponent < 0.2 : covered.redComponent > 0.9)
      #expect(
        gutter.redComponent > gutter.greenComponent + 0.5,
        "The bottom cover must not obscure the scrollbar gutter")
    }
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
      scroll.scrollerInsets.top == 0 && scroll.scrollerInsets.bottom == 0,
      "Content clearance must not shorten the full-height scrollbar track")

    for height: CGFloat in [172, 320, 120] {
      state.composer = height
      try await settle(host.view)
      #expect(scroll.documentView === document)
      #expect(abs(scroll.contentView.bounds.height - 700) < 1)
      #expect(
        abs(document.bounds.maxY - scroll.documentVisibleRect.maxY) < 2,
        "Tail must stay pinned when the composer grows or shrinks")
      let text = try #require(find(NSTextView.self, in: document))
      let textRect = text.convert(text.bounds, to: document)
      #expect(
        scroll.documentVisibleRect.maxY - textRect.maxY >= height + 24,
        "The final text and message actions must stop above the floating composer")
    }

    state.paragraphs += 4
    try await settle(host.view)
    #expect(abs(document.bounds.maxY - scroll.documentVisibleRect.maxY) < 2)
    for height: CGFloat in [520, 820, 700] {
      window.setContentSize(NSSize(width: 820, height: height))
      try await settle(host.view)
      #expect(abs(scroll.contentView.bounds.height - height) < 1)
      #expect(abs(document.bounds.maxY - scroll.documentVisibleRect.maxY) < 2)
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
    #expect(document.bounds.maxY - scroll.documentVisibleRect.maxY > 100)
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
  var transcript: StudioState.NativeTranscript {
    let text = (1...paragraphs).map {
      "Paragraph \($0). Reading position and selection stay intact while the native composer grows and the answer continues streaming."
    }.joined(separator: "\n\n")
    let bytes = try! JSONSerialization.data(withJSONObject: [
      "available": true, "notice": "",
      "messages": [
        [
          "id": "reply", "role": "assistant", "text": text, "reasoning": "",
          "model": "fixture", "streaming": false, "stopped": false, "error": "",
          "incomplete": false,
        ]
      ],
    ])
    return try! JSONDecoder().decode(StudioState.NativeTranscript.self, from: bytes)
  }
}

private struct TranscriptLayoutView: View {
  @Bindable var state: TranscriptLayoutFixture
  var body: some View {
    NativeStudioTranscript(
      transcript: state.transcript, columnWidth: 760, composerHeight: state.composer
    )
    .overlay(alignment: .bottom) {
      Color.gray.frame(width: 760, height: state.composer).allowsHitTesting(false)
    }
  }
}
