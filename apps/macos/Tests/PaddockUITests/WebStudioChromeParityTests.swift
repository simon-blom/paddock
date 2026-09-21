import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockNativeMarkdown
@testable import PaddockUI

@Suite("Web Studio chrome parity", .serialized) @MainActor
struct WebStudioChromeParityTests {
  @Test func actualCompareThinkingHeaderAcceptsPointerClicks() async throws {
    _ = NSApplication.shared
    let rows: [[String: Any]] = (0..<2).map { i in
      [
        "id": "click-\(i)", "group": "compare", "role": "assistant", "model": "model-\(i)",
        "text": "Answer \(i)",
        "reasoning": (0..<24).map { "Reasoning to expand in lane \(i), step \($0)." }.joined(
          separator: "\n\n"), "streaming": false,
        "stopped": false, "error": "", "incomplete": false,
      ]
    }
    let transcript = try JSONDecoder().decode(
      StudioState.NativeTranscript.self,
      from: JSONSerialization.data(withJSONObject: [
        "available": true, "notice": "", "messages": rows,
      ]))
    let host = NSHostingController(
      rootView: NativeStudioTranscript(
        transcript: transcript, columnWidth: 944, composerHeight: 120
      )
      .frame(width: 1000, height: 700))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 1000, height: 700), styleMask: [.borderless],
      backing: .buffered, defer: false)
    host.sizingOptions = []
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setContentSize(CGSize(width: 1000, height: 700))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for _ in 0..<100 {
      host.view.layoutSubtreeIfNeeded()
      if texts(host.view).contains(where: { $0.string == "Answer 0" }) { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    let text = try #require(texts(host.view).first { $0.string == "Answer 0" })
    let scroll = try #require(find(NSScrollView.self, host.view))
    let originalOffset = scroll.contentView.bounds.minY
    let frame = text.convert(text.bounds, to: host.view)
    // Header immediately precedes this short answer: 12pt content gap and
    // half of its 28pt header. Unlike AXPress this exercises the entire hit path.
    let point = host.view.convert(
      NSPoint(
        x: frame.midX,
        y: host.view.isFlipped ? frame.minY - 26 : frame.maxY + 26), to: nil)
    for opens in [true, false, true] {
      for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
        window.sendEvent(
          try #require(
            NSEvent.mouseEvent(
              with: type, location: point, modifierFlags: [],
              timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
              context: nil,
              eventNumber: 1, clickCount: 1, pressure: 1)))
      }
      for _ in 0..<100 {
        host.view.layoutSubtreeIfNeeded()
        if texts(host.view).contains(where: { $0.string.contains("Reasoning to expand in lane 0") })
          == opens
        {
          break
        }
        try await Task.sleep(for: .milliseconds(10))
      }
      try await Task.sleep(for: .milliseconds(150))
      host.view.layoutSubtreeIfNeeded()
      #expect(
        texts(host.view).contains(where: { $0.string.contains("Reasoning to expand in lane 0") })
          == opens)
      #expect(
        abs(scroll.contentView.bounds.minY - originalOffset) < 1,
        "Reading a thought must not jump to the end of Compare")
    }
  }

  @Test func thinkingHeaderAcceptsMouseClicksInsideTheConversationSelectionHost() async throws {
    _ = NSApplication.shared
    let message = try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.self,
      from: Data(
        #"{"id":"click","role":"assistant","model":"Qwen","text":"Answer","reasoning":"The reasoning must open with the mouse.","streaming":false,"stopped":false,"error":"","incomplete":false}"#
          .utf8))
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: ConversationSelectionSurface(items: []) {
          VStack {
            NativeThinkingBlock(message: message)
            Spacer()
          }.padding(10)
        }.environment(\.colorScheme, dark ? .dark : .light))
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 400, height: 400),
        styleMask: [.borderless], backing: .buffered, defer: false)
      host.sizingOptions = []
      window.isReleasedWhenClosed = false
      window.contentViewController = host
      window.setContentSize(CGSize(width: 400, height: 400))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      try await Task.sleep(for: .milliseconds(100))
      host.view.layoutSubtreeIfNeeded()
      #expect(texts(host.view).isEmpty)
      // Click real window coordinates through AppKit/SwiftUI hit testing,
      // not accessibilityPerformPress (which bypasses a blocking overlay).
      let selection = try #require(find(ConversationSelectionHost.self, host.view))
      for opens in [true, false, true] {
        let point = host.view.convert(
          NSPoint(x: 180, y: host.view.isFlipped ? 24 : host.view.bounds.height - 24), to: nil)
        for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
          let event = try #require(
            NSEvent.mouseEvent(
              with: type, location: point,
              modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
              windowNumber: window.windowNumber, context: nil, eventNumber: 1, clickCount: 1,
              pressure: 1))
          window.sendEvent(event)
        }
        for _ in 0..<100 {
          host.view.layoutSubtreeIfNeeded()
          if texts(host.view).contains(where: { $0.string.contains("reasoning must open") })
            == opens
          {
            break
          }
          try await Task.sleep(for: .milliseconds(10))
        }
        #expect(
          texts(host.view).contains(where: { $0.string.contains("reasoning must open") }) == opens)
      }
      // AppKit also bubbles control events to the hosting responder. A direct
      // forward is not a full SwiftUI gesture dispatch, but must never start a
      // text selection at the nearest paragraph (the original regression).
      window.makeFirstResponder(nil)
      let point = host.view.convert(
        NSPoint(x: 180, y: host.view.isFlipped ? 24 : host.view.bounds.height - 24), to: nil)
      for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
        let event = try #require(
          NSEvent.mouseEvent(
            with: type, location: point, modifierFlags: [],
            timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
            context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
        if type == .leftMouseDown {
          selection.mouseDown(with: event)
          #expect(!selection.selecting, "Control presses must not be captured as text selection")
        } else {
          selection.mouseUp(with: event)
        }
      }
    }
  }

  @Test func compareKeepsStreamingClearanceWithoutExtraSpaceBelowTheFooter() async throws {
    _ = NSApplication.shared
    for streaming in [true, false] {
      let message = try NativeMessageSample.message(
        text: "I'll create a clean, modern hero section page for you.", streaming: streaming)
      let probe = NSView()
      let host = NSHostingController(
        rootView: VStack {
          NativeStudioMessage(message: message, inLane: true)
            .fixedSize(horizontal: false, vertical: true).background(FrameProbe(view: probe))
          Spacer()
        }.frame(width: 350, height: 800, alignment: .topLeading))
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 350, height: 800),
        styleMask: [.borderless], backing: .buffered, defer: false)
      host.sizingOptions = []
      window.isReleasedWhenClosed = false
      window.contentViewController = host
      window.setContentSize(CGSize(width: 350, height: 800))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for _ in 0..<100 {
        host.view.layoutSubtreeIfNeeded()
        if texts(host.view).contains(where: { $0.string.contains("clean, modern hero") }) { break }
        try await Task.sleep(for: .milliseconds(20))
      }
      try await Task.sleep(for: .milliseconds(150))
      host.view.layoutSubtreeIfNeeded()
      let bounds = probe.convert(probe.bounds, to: host.view)
      #expect(bounds.height >= 28)
      let text = try #require(texts(host.view).first { $0.string.contains("clean, modern hero") })
      let frame = text.convert(text.bounds, to: host.view)
      let clearance = host.view.isFlipped ? bounds.maxY - frame.maxY : frame.minY - bounds.minY
      if streaming {
        #expect(
          clearance >= 27 && clearance <= 33,
          "Streaming prose needs the web's bottom margin, not 4pt: \(clearance)")
      } else {
        let footer = NSHostingController(rootView: NativeMessageFooter(message: message))
        let footerHeight = footer.sizeThatFits(in: CGSize(width: 350, height: 800)).height
        #expect(
          abs(clearance - (4 + footerHeight)) <= 1,
          "Only the 4pt content gap and footer belong here, not another 28pt bottom margin: \(clearance)"
        )
      }
    }
  }

  private func texts(_ view: NSView) -> [NSTextView] {
    ((view as? NSTextView).map { [$0] } ?? []) + view.subviews.flatMap(texts)
  }
  private func find<T: NSView>(_ type: T.Type, _ root: NSView) -> T? {
    (root as? T) ?? root.subviews.lazy.compactMap { find(type, $0) }.first
  }
  private struct FrameProbe: NSViewRepresentable {
    let view: NSView
    func makeNSView(context: Context) -> NSView { view }
    func updateNSView(_ nsView: NSView, context: Context) {}
  }

  @Test func previewHeaderIsOneCompactRow() throws {
    _ = NSApplication.shared
    let versions = try JSONDecoder().decode(
      [StudioWorkspace.ArtifactContent.Version].self,
      from: Data(#"[{"seq":1,"op":"create","bytes":10},{"seq":2,"op":"edit","bytes":20}]"#.utf8))
    for dark in [false, true] {
      for width: CGFloat in [340, 700] {
        let header = NativeArtifactHeader(
          writer: ("Qwen 3.8 27B", "Qwen"), model: "qwen", title: "A complete hero page",
          source: .constant(false), version: .constant(0),
          versions: width == 340 ? Array(versions.prefix(1)) : versions,
          dirty: width > 340, saving: false, available: true, copied: false,
          save: {}, revert: {}, copy: {}, download: {}, close: {})
        let host = NSHostingController(
          rootView: header.environment(\.colorScheme, dark ? .dark : .light))
        let fit = host.sizeThatFits(in: CGSize(width: width, height: 1000))
        #expect(fit.width <= width + 1)
        #expect(fit.height <= 44, "ArtifactPane uses one header row, not three toolbars: \(fit)")
      }
    }
  }

  @Test func compactToolCardsAndArgumentsMatchWebShape() throws {
    _ = NSApplication.shared
    let call = try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.ToolCall.self,
      from: Data(
        #"{"id":"tool","name":"artifacts__artifact_create","server":"artifacts","arguments":"{}","output":"ok","status":"completed","error":"","artifactId":"art_012345abcdef"}"#
          .utf8))
    let host = NSHostingController(
      rootView: NativeToolCallView(call: call, workspace: nil, target: nil))
    for width: CGFloat in [240, 760] {
      let fit = host.sizeThatFits(in: CGSize(width: width, height: 1000))
      #expect(fit.width <= width + 1)
      #expect(fit.height <= 34, "ToolCall.vue uses a compact bordered 7px/10px disclosure: \(fit)")
    }
    #expect(
      ToolCallPresentation.arguments(#"{"body":"<html>\nhello\n</html>","title":"Page"}"#)
        == "body:\n  <html>\n  hello\n  </html>\ntitle: Page")
    #expect(
      ToolCallPresentation.output(#"[{"type":"text","text":"Created art_012345abcdef"}]"#)
        == "Created art_012345abcdef")
    #expect(ToolCallPresentation.arguments("partial") == "partial")
  }

  @Test func waitingHasNoEmptyMarkdownOrLargeSpinnerRegion() async throws {
    _ = NSApplication.shared
    let message = try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.self,
      from: Data(
        #"{"id":"waiting","role":"assistant","model":"Qwen","text":"","reasoning":"","streaming":true,"stopped":false,"error":"","incomplete":false}"#
          .utf8))
    let host = NSHostingController(rootView: NativeStudioMessage(message: message))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 600, height: 400),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(550))
    host.view.layoutSubtreeIfNeeded()
    let fit = host.sizeThatFits(in: CGSize(width: 600, height: 1000))
    #expect(fit.height >= 28 && fit.height <= 65, "A 28pt pill below the model stamp: \(fit)")
    func textViews(_ view: NSView) -> [NSTextView] {
      (view as? NSTextView).map { [$0] } ?? view.subviews.flatMap(textViews)
    }
    #expect(
      textViews(host.view).isEmpty, "Do not mount an empty Markdown text container while waiting")
  }
}
