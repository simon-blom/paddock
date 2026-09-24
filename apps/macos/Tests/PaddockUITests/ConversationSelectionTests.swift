import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockNativeMarkdown
@testable import PaddockUI

@Suite("Conversation-wide native selection", .serialized) @MainActor
struct ConversationSelectionTests {
  @Test func actualDragSpansUserReasoningAnswerCodeAndAnotherMessage() async throws {
    let fixture = try await mount()
    defer { fixture.window.close() }
    let root = fixture.selection
    let views = texts(root).filter { !$0.isEditable && !hasTextAncestor($0, within: root) }
    let first = try #require(views.first { $0.string.contains("User question") })
    let last = try #require(views.first { $0.string.contains("Last answer") })
    let start = point(first, character: 0)
    let end = point(last, character: (last.string as NSString).length)
    // Hit testing, not just setSelectedRange: the old implementation selected
    // only the first NSTextView, even though range-only unit checks passed.
    #expect(
      root.hitTest(
        root.superview!.convert(start, from: nil),
        event: mouse(.leftMouseDown, point: start, window: fixture.window)) === root)
    root.mouseDown(with: mouse(.leftMouseDown, point: start, window: fixture.window))
    root.mouseDragged(with: mouse(.leftMouseDragged, point: end, window: fixture.window))
    root.mouseUp(with: mouse(.leftMouseUp, point: end, window: fixture.window))
    root.copy(nil)
    let copy = try #require(NSPasteboard.general.string(forType: .string))
    #expect(copy.contains("User question"))
    #expect(copy.contains("Live reasoning"))
    #expect(copy.contains("Before code"))
    #expect(copy.contains("let answer = 42"))
    #expect(copy.contains("After code"))
    #expect(copy.contains("Last answer"))
    #expect(!copy.contains("\u{FFFC}"))
    #expect(views.allSatisfy { $0.selectedRange().length > 0 })
    // Reverse drags must produce the same ordered text, not reverse messages.
    root.mouseDown(with: mouse(.leftMouseDown, point: end, window: fixture.window))
    root.mouseDragged(with: mouse(.leftMouseDragged, point: start, window: fixture.window))
    root.mouseUp(with: mouse(.leftMouseUp, point: start, window: fixture.window))
    root.copy(nil)
    #expect(NSPasteboard.general.string(forType: .string) == copy)
  }

  @Test func hoverHitTestingLeavesTextWithItsNativeOwner() async throws {
    let fixture = try await mount()
    defer { fixture.window.close() }
    let root = fixture.selection
    let text = try #require(texts(root).first { $0.string.contains("Before code") })
    let start = point(text, character: 0)
    let local = root.superview!.convert(start, from: nil)
    // SwiftUI asks for hit targets during hover/layout, including without an
    // active event. Neither path may resolve a TextKit insertion point or
    // redirect an embedded Markdown attachment to the selection responder.
    #expect(root.hitTest(local, event: nil) === text)
    for type: NSEvent.EventType in [.mouseMoved, .leftMouseUp] {
      let event = try #require(
        NSEvent.mouseEvent(
          with: type, location: start, modifierFlags: [],
          timestamp: 1, windowNumber: fixture.window.windowNumber, context: nil,
          eventNumber: 1, clickCount: 0, pressure: 0))
      #expect(root.hitTest(local, event: event) === text)
    }
    let wheel = try #require(
      CGEvent(
        scrollWheelEvent2Source: nil, units: .pixel,
        wheelCount: 1, wheel1: 120, wheel2: 0, wheel3: 0))
    #expect(root.hitTest(local, event: NSEvent(cgEvent: wheel)) === text)
  }

  @Test func selectAllIncludesUnmountedMessagesButNotFoldedThinkingOrComposer() async throws {
    let f = try await mount()
    defer { f.window.close() }
    f.selection.items += [
      .init(id: "offscreen", text: "Older message outside the lazy viewport"),
      .init(id: "folded", text: "HIDDEN REASONING", includeWhenUnmounted: false),
    ]
    f.selection.selectAll(nil)
    f.selection.copy(nil)
    let copied = try #require(NSPasteboard.general.string(forType: .string))
    #expect(copied.contains("Older message outside the lazy viewport"))
    #expect(copied.contains("let answer = 42"))
    #expect(!copied.contains("HIDDEN REASONING"))
    #expect(!copied.contains("Unsent composer draft"))
    #expect(!copied.contains("**"))
    let draft = try #require(texts(f.host).first { $0.isEditable })
    #expect(draft.selectedRange().length == 0)
  }

  @Test func pointerHitTestingNeverMeasuresTextAndLinksReachTheirTextOwner() async throws {
    let f = try await mount()
    defer { f.window.close() }
    let storage = NSTextStorage(string: "A linked word")
    let manager = HitTestLayoutManager()
    storage.addLayoutManager(manager)
    let container = NSTextContainer(containerSize: NSSize(width: 300, height: 80))
    manager.addTextContainer(container)
    let text = LinkPressTextView(
      frame: NSRect(x: 30, y: 30, width: 300, height: 80), textContainer: container)
    text.isEditable = false
    text.isSelectable = true
    text.identifier = .init("link-fixture")
    f.selection.items.append(.init(id: "link-fixture", text: storage.string))
    storage.addAttribute(
      .link, value: URL(string: "https://example.com")!,
      range: NSRange(location: 0, length: storage.length))
    f.selection.addSubview(text)
    let location = point(text, character: 0)
    let local = f.selection.superview!.convert(location, from: nil)
    let before = manager.measurements
    // A hover/layout query can observe an old currentEvent of any type.
    // Even a mouse-down hit test must not measure the attachment tree.
    for type: NSEvent.EventType in [.leftMouseDown, .leftMouseDragged, .rightMouseDown] {
      let target = f.selection.hitTest(local, event: mouse(type, point: location, window: f.window))
      #expect(target === f.selection || target === text)
      #expect(manager.measurements == before)
    }
    f.selection.mouseDown(with: mouse(.leftMouseDown, point: location, window: f.window))
    #expect(text.presses == 1)
    #expect(!f.selection.selecting)
  }

  @Test func dragRetainsLogicalOrderWhenTheAnchorLosesItsHostingAncestor() async throws {
    let f = try await mount()
    defer { f.window.close() }
    let first = try #require(texts(f.host).first { $0.string == "Live reasoning" })
    let last = try #require(texts(f.host).first { $0.string == "Last answer" })
    let start = point(first, character: 0)
    let end = point(last, character: (last.string as NSString).length)
    f.selection.mouseDown(with: mouse(.leftMouseDown, point: start, window: f.window))
    // Simulate lazy unmounting: the retained NSTextView no longer has the
    // SelectionHostingView ancestor that supplied its stable message ID.
    first.removeFromSuperview()
    f.selection.mouseDragged(with: mouse(.leftMouseDragged, point: end, window: f.window))
    f.selection.mouseUp(with: mouse(.leftMouseUp, point: end, window: f.window))
    f.selection.copy(nil)
    let copied = try #require(NSPasteboard.general.string(forType: .string))
    #expect(copied.hasPrefix("Live reasoning"))
    #expect(copied.contains("let answer = 42"))
    #expect(copied.hasSuffix("Last answer"))
    #expect(!copied.contains("User question"))
  }

  @Test func draggingInsideCodeCopiesOnlyTheChosenCodeSubstring() async throws {
    let f = try await mount()
    defer { f.window.close() }
    let code = try #require(
      texts(f.host).first { $0.string == "let answer = 42" || $0.string == "let answer = 42\n" })
    let start = point(code, character: 4)
    let end = point(code, character: 10)
    f.selection.mouseDown(with: mouse(.leftMouseDown, point: start, window: f.window))
    f.selection.mouseDragged(with: mouse(.leftMouseDragged, point: end, window: f.window))
    f.selection.mouseUp(with: mouse(.leftMouseUp, point: end, window: f.window))
    f.selection.copy(nil)
    #expect(NSPasteboard.general.string(forType: .string) == "answer")
  }
  @Test func actualCompareLanesShareSelectionWithoutIncludingTheComposer() async throws {
    let rows: [[String: Any]] = ["Left answer", "Right answer"].enumerated().map { i, text in
      [
        "id": "lane-\(i)", "role": "assistant", "model": "model-\(i)", "text": text,
        "reasoning": "", "streaming": false, "stopped": false, "error": "", "incomplete": false,
        "group": "compare",
      ]
    }
    let transcript = try JSONDecoder().decode(
      StudioState.NativeTranscript.self,
      from: JSONSerialization.data(withJSONObject: [
        "available": true, "notice": "", "messages": rows,
      ]))
    let host = NSHostingView(
      rootView: NativeStudioTranscript(
        transcript: transcript, columnWidth: 944, composerHeight: 120,
        composer: AnyView(
          VStack {
            TextField("Draft", text: .constant("Do not select the draft"))
            NativeSelectableText("Do not select this audio preview")
          }.frame(height: 120))
      ).frame(width: 1000, height: 700))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1000, height: 700), styleMask: [.borderless],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for _ in 0..<100 {
      host.layoutSubtreeIfNeeded()
      if texts(host).contains(where: { $0.string.contains("Right answer") }) { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    try await Task.sleep(for: .milliseconds(100))
    let selection = try #require(find(ConversationSelectionHost.self, host))
    let left = try #require(texts(host).first { $0.string == "Left answer" })
    let right = try #require(texts(host).first { $0.string == "Right answer" })
    let start = point(left, character: 0)
    let end = point(right, character: (right.string as NSString).length)
    selection.mouseDown(with: mouse(.leftMouseDown, point: start, window: window))
    selection.mouseDragged(with: mouse(.leftMouseDragged, point: end, window: window))
    selection.mouseUp(with: mouse(.leftMouseUp, point: end, window: window))
    selection.copy(nil)
    #expect(NSPasteboard.general.string(forType: .string) == "Left answer\n\nRight answer")
    selection.selectAll(nil)
    selection.copy(nil)
    #expect(NSPasteboard.general.string(forType: .string) == "Left answer\n\nRight answer")
    let preview = try #require(
      texts(host).first { $0.string == "Do not select this audio preview" })
    let previewPoint = point(preview, character: 4)
    let previewEvent = mouse(.leftMouseDown, point: previewPoint, window: window)
    let hitPoint = selection.superview?.convert(previewPoint, from: nil) ?? previewPoint
    #expect(selection.hitTest(hitPoint, event: previewEvent) === preview)
  }
  @Test func unmountedMarkdownCopyIsCompleteAndRenderedOffMain() async throws {
    let f = try await mount()
    defer { f.window.close() }
    f.selection.items += (0..<1000).map {
      .init(id: "off-\($0)", text: "**Older \($0)** with `code`", markdown: true)
    }
    f.selection.selectAll(nil)
    f.selection.copy(nil)
    for _ in 0..<100 {
      if NSPasteboard.general.string(forType: .string)?.contains("Older 999 with code") == true {
        break
      }
      try await Task.sleep(for: .milliseconds(20))
    }
    let text = try #require(NSPasteboard.general.string(forType: .string))
    #expect(text.contains("Older 0 with code") && text.contains("Older 999 with code"))
    #expect(!text.contains("**"))
  }

  private func mount() async throws -> (
    host: NSView, selection: ConversationSelectionHost, window: NSWindow
  ) {
    _ = NSApplication.shared
    let root = NSHostingView(rootView: Fixture())
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 760, height: 900), styleMask: [.borderless],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = root
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    for _ in 0..<100 {
      root.layoutSubtreeIfNeeded()
      if texts(root).contains(where: { $0.string.contains("After code") }) { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    try await Task.sleep(for: .milliseconds(100))
    root.layoutSubtreeIfNeeded()
    return (root, try #require(find(ConversationSelectionHost.self, root)), window)
  }
  private struct Fixture: View {
    var body: some View {
      VStack {
        ConversationSelectionSurface(items: [
          .init(id: "user", text: "User question"),
          .init(id: "reasoning", text: "Live reasoning", includeWhenUnmounted: false),
          .init(id: "answer", text: "Before code\nlet answer = 42\nAfter code"),
          .init(id: "last", text: "Last answer"),
        ]) {
          ScrollView {
            VStack(alignment: .leading, spacing: 24) {
              NativeSelectableText("User question").environment(\.conversationTextID, "user")
              NativeMarkdown("Live reasoning").environment(\.conversationTextID, "reasoning")
              NativeMarkdown("Before **code**\n\n```swift\nlet answer = 42\n```\n\nAfter code")
                .environment(\.conversationTextID, "answer")
              NativeMarkdown("Last answer").environment(\.conversationTextID, "last")
            }.padding(24)
          }
        }
        TextEditor(text: .constant("Unsent composer draft")).frame(height: 70)
      }.frame(width: 760, height: 900)
    }
  }
  private func mouse(_ type: NSEvent.EventType, point: NSPoint, window: NSWindow) -> NSEvent {
    NSEvent.mouseEvent(
      with: type, location: point, modifierFlags: [], timestamp: 1,
      windowNumber: window.windowNumber, context: nil, eventNumber: 1, clickCount: 1, pressure: 1)!
  }
  private func point(_ view: NSTextView, character: Int) -> NSPoint {
    let manager = view.layoutManager!
    let container = view.textContainer!
    manager.ensureLayout(for: container)
    let n = (view.string as NSString).length
    let index = min(character, max(0, n - 1))
    let glyph = manager.glyphIndexForCharacter(at: index)
    let rect = manager.boundingRect(
      forGlyphRange: NSRange(location: glyph, length: 1), in: container)
    return view.convert(
      NSPoint(
        x: (character == n ? rect.maxX : rect.minX) + view.textContainerOrigin.x + 0.1,
        y: rect.midY + view.textContainerOrigin.y), to: nil)
  }
  private func texts(_ root: NSView) -> [NSTextView] {
    ((root as? NSTextView).map { [$0] } ?? []) + root.subviews.flatMap(texts)
  }
  private func find<T: NSView>(_ type: T.Type, _ root: NSView) -> T? {
    (root as? T) ?? root.subviews.lazy.compactMap { find(type, $0) }.first
  }
  private func hasTextAncestor(_ view: NSView, within root: NSView) -> Bool {
    var p = view.superview
    while let parent = p, parent !== root {
      if parent is NSTextView { return true }
      p = parent.superview
    }
    return false
  }
}

private final class HitTestLayoutManager: NSLayoutManager {
  var measurements = 0
  override func ensureLayout(for container: NSTextContainer) {
    measurements += 1
    super.ensureLayout(for: container)
  }
}

@MainActor private final class LinkPressTextView: NSTextView {
  var presses = 0
  override func mouseDown(with event: NSEvent) { presses += 1 }
}
