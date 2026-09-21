import AppKit
import Observation
import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Content-sized native tool results", .serialized) @MainActor
struct NativeToolCodeTests {
  @Test func shortResultsHugTextAndLongResultsScrollInBothThemes() async throws {
    for dark in [false, true] {
      for width: CGFloat in [240, 760] {
        let state = ToolCodeFixture()
        let host = NSHostingController(
          rootView: ToolCodeFixtureView(state: state)
            .environment(\.colorScheme, dark ? .dark : .light))
        let window = mount(host, width: width)
        defer { window.close() }
        for value in ["Created artifact.", Self.result, Self.longResult, "Done"] {
          state.text = value
          try await Task.sleep(for: .milliseconds(60))
          host.view.layoutSubtreeIfNeeded()
          let scroll = try #require(
            views(host.view).compactMap { $0 as? NativeToolCodeScrollView }.first)
          let text = scroll.textView
          #expect(text.string == value && text.identifier?.rawValue == "tool/output")
          #expect(text.textContainerOrigin == NSPoint(x: 11, y: 9))
          #expect(scroll.bounds.height <= 260 && abs(scroll.bounds.width - width) < 1)
          if value == Self.longResult {
            #expect(scroll.bounds.height == 260 && scroll.hasVerticalScroller)
            #expect(scroll.verticalScroller is PaddockScroller)
            #expect(text.frame.height > scroll.contentSize.height)
            text.scrollRangeToVisible(NSRange(location: text.string.utf16.count - 1, length: 1))
            #expect(scroll.contentView.bounds.minY > 0, "The final result line remains reachable")
          } else {
            #expect(!scroll.hasVerticalScroller)
            #expect(
              abs(scroll.bounds.height - text.frame.height) < 1,
              "No spare space above/below a short result: \(scroll.bounds) vs \(text.frame)")
            let layout = try #require(text.layoutManager)
            let container = try #require(text.textContainer)
            #expect(
              abs(scroll.bounds.height - ceil(layout.usedRect(for: container).height) - 18) < 1)
            #expect(
              scroll.contentView.bounds.minY == 0,
              "Shrinking from long output clamps its scroll offset")
          }
        }
      }
    }
  }

  @Test func resizeAndStreamingKeepTheSameTextViewAndSelection() async throws {
    let state = ToolCodeFixture()
    state.text = Self.result
    let host = NSHostingController(rootView: ToolCodeFixtureView(state: state))
    let window = mount(host, width: 760)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(60))
    host.view.layoutSubtreeIfNeeded()
    let scroll = try #require(views(host.view).compactMap { $0 as? NativeToolCodeScrollView }.first)
    let original = scroll.textView
    let selection = NSRange(location: 0, length: 7)
    original.setSelectedRange(selection)
    for (width, value) in [
      (CGFloat(260), Self.result), (260, Self.result + "\n" + Self.longResult), (760, Self.result),
    ] {
      state.text = value
      window.setContentSize(NSSize(width: width, height: 650))
      try await Task.sleep(for: .milliseconds(80))
      host.view.layoutSubtreeIfNeeded()
      #expect(scroll.textView === original && original.selectedRange() == selection)
      #expect(original.string == value)
      #expect(abs(scroll.bounds.width - width) < 1 && scroll.bounds.height <= 260)
      #expect(original.isSelectable && !original.isEditable)
    }
  }

  @Test func expandedArtifactCardHasNoResultGapInRegularAndCompareLayouts() async throws {
    let call: [String: Any] = [
      "id": "create", "name": "artifact_create", "server": "artifacts",
      "arguments": "{}", "output": Self.result, "status": "completed", "error": "",
    ]
    for compare in [false, true] {
      let raw: [String: Any] = [
        "id": "answer", "role": "assistant", "model": "", "text": "",
        "reasoning": "", "streaming": false, "stopped": false, "incomplete": false,
        "error": "", "toolCalls": [call],
      ]
      let message = try JSONDecoder().decode(
        StudioState.NativeTranscript.Message.self,
        from: JSONSerialization.data(withJSONObject: raw))
      let host = NSHostingController(
        rootView: VStack {
          NativeStudioMessage(message: message, inLane: compare)
          Spacer(minLength: 0)
        }.padding(12))
      let window = mount(host, width: compare ? 340 : 760)
      defer { window.close() }
      try await Task.sleep(for: .milliseconds(80))
      host.view.layoutSubtreeIfNeeded()
      // The empty reply has no model-header row. Its compact tool disclosure
      // is the first control. Exercise real pointer delivery, not AXPress.
      let point = host.view.convert(
        NSPoint(x: 100, y: host.view.isFlipped ? 26 : host.view.bounds.height - 26), to: nil)
      for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
        window.sendEvent(
          try #require(
            NSEvent.mouseEvent(
              with: type, location: point, modifierFlags: [],
              timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
              context: nil, eventNumber: 1, clickCount: 1, pressure: 1)))
      }
      try await Task.sleep(for: .milliseconds(80))
      host.view.layoutSubtreeIfNeeded()
      let result = try #require(
        views(host.view).compactMap { $0 as? NativeToolCodeScrollView }
          .first { $0.textView.identifier?.rawValue == "create/output" })
      #expect(result.textView.string == Self.result)
      #expect(abs(result.bounds.height - result.textView.frame.height) < 1)
      #expect(result.bounds.height < 150, "The real card must not stretch short results to the cap")
    }
  }

  @Test func conversationSelectionIncludesTheEntireScrollableResult() async throws {
    let host = NSHostingController(
      rootView: ConversationSelectionSurface(items: [
        .init(id: "before", text: "Before result"),
        .init(id: "tool/output", text: Self.longResult, includeWhenUnmounted: false),
        .init(id: "after", text: "After result"),
      ]) {
        VStack {
          NativeSelectableText("Before result").environment(\.conversationTextID, "before")
          NativeToolCode(text: Self.longResult, textID: "tool/output")
          NativeSelectableText("After result").environment(\.conversationTextID, "after")
          Spacer(minLength: 0)
        }
      })
    let window = mount(host, width: 420)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    let selection = try #require(
      views(host.view).compactMap { $0 as? ConversationSelectionHost }.first)
    selection.selectAll(nil)
    let text = views(host.view).compactMap { $0 as? NSTextView }
    #expect(text.count == 3)
    for view in text {
      #expect(
        view.selectedRange() == NSRange(location: 0, length: view.string.utf16.count),
        "Clipping long tool results must not clip conversation-wide selection")
    }
  }

  private static let result =
    "Created art_fixture (html, 253 lines) and opened it in the side panel. Edit it with artifact_update; do not repeat the content in your reply."
  private static let longResult = (0..<200).map {
    "Result line \($0): keep the complete content selectable."
  }.joined(separator: "\n")
  private func views(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(views) }
  private func mount<Content: View>(_ host: NSHostingController<Content>, width: CGFloat)
    -> NSWindow
  {
    _ = NSApplication.shared
    host.sizingOptions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: width, height: 650),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setContentSize(NSSize(width: width, height: 650))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    return window
  }
}

@MainActor @Observable private final class ToolCodeFixture { var text = "" }

private struct ToolCodeFixtureView: View {
  @Bindable var state: ToolCodeFixture
  var body: some View {
    VStack {
      NativeToolCode(text: state.text, textID: "tool/output")
      Spacer(minLength: 0)
    }
  }
}
