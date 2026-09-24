import AppKit
import PaddockStudio
import SwiftUI
import Testing
import WebKit

@testable import PaddockUI

@Suite("Native comparison rendering", .serialized) @MainActor
struct NativeCompareRenderingTests {
  @Test func stepsKeepLaneOrderAndTheWholeGroupAnchor() throws {
    let transcript = try fixture(count: 4)
    #expect(transcript.hasComparisons)
    #expect(transcript.blocks.map(\.id) == ["question", "lane-0"])
    #expect(transcript.blocks.map(\.comparison) == [false, true])
    #expect(transcript.blocks[1].messages.map(\.id) == (0..<4).map { "lane-\($0)" })
    #expect(transcript.blocks[1].messages[0].streaming)
    #expect(transcript.blocks[1].messages[1].chrome?.vendor == "Meta")
    #expect(transcript.blocks[1].messages[1].chrome?.footer == "120 tokens · 60 tok/s · 5.6s")
  }

  @Test func columnsNeverWrapAndShareTheComposerBounds() {
    for count in 2...4 {
      for available: CGFloat in [320, 600, 1000, 1600] {
        let column = StudioColumnLayout.resolve(
          available: available, viewport: nil, comparison: true)
        #expect(column.width <= available)
        #expect(column.midX == available / 2)
        let lane = NativeCompareLayout.laneWidth(available: column.width, count: count)
        #expect(lane >= 240)
        #expect(lane * CGFloat(count) + 12 * CGFloat(count - 1) >= column.width - 0.01)
      }
    }
  }

  @Test func lanesUseNativeTextInBothAppearancesWithoutWebKit() async throws {
    _ = NSApplication.shared
    let transcript = try fixture(count: 4)
    for dark in [false, true] {
      let host = NSHostingController(
        rootView:
          NativeCompareBlock(block: transcript.blocks[1], transcript: transcript, width: 620)
          .frame(width: 620).environment(\.colorScheme, dark ? .dark : .light))
      let fit = host.sizeThatFits(in: CGSize(width: 620, height: 1200))
      #expect(fit.width <= 621)
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 620, height: fit.height),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      host.sizingOptions = []
      window.contentViewController = host
      window.setContentSize(CGSize(width: 620, height: fit.height))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for _ in 0..<100 {
        host.view.layoutSubtreeIfNeeded()
        if descendants(host.view).compactMap({ $0 as? NSTextView })
          .filter({ $0.string.contains("Selectable") }).count >= 4
        {
          break
        }
        try await Task.sleep(for: .milliseconds(25))
      }
      let rendered = host.sizeThatFits(in: CGSize(width: 620, height: 1200))
      #expect(
        rendered.height > 120 && rendered.height < 1200,
        "Loaded Markdown should set the lane height, got \(rendered.height)")
      window.setContentSize(CGSize(width: 620, height: rendered.height))
      let views = descendants(host.view)
      #expect(!views.contains { $0 is WKWebView })
      let text = views.compactMap { $0 as? NSTextView }
      #expect(text.count >= 4)
      #expect(text.allSatisfy { $0.isSelectable })
      let horizontal = views.compactMap { $0 as? NSScrollView }.first {
        ($0.documentView?.frame.width ?? 0) > 900
      }
      #expect(horizontal != nil, "Four lanes must overflow horizontally, not squeeze or wrap")
      let sibling = try #require(text.first { $0.string.contains("Lane 1") })
      let selected = NSRange(location: 0, length: min(6, (sibling.string as NSString).length))
      sibling.setSelectedRange(selected)
      horizontal?.contentView.scroll(to: NSPoint(x: 120, y: 0))
      if let horizontal { horizontal.reflectScrolledClipView(horizontal.contentView) }
      let updated = try fixture(
        count: 4, delta: "\n\nAdditional streamed tokens in only the first lane.")
      host.rootView = NativeCompareBlock(block: updated.blocks[1], transcript: updated, width: 620)
        .frame(width: 620).environment(\.colorScheme, dark ? .dark : .light)
      try await Task.sleep(for: .milliseconds(150))
      host.view.layoutSubtreeIfNeeded()
      let retained = descendants(host.view).compactMap { $0 as? NSTextView }.first {
        $0.string.contains("Lane 1")
      }
      #expect(retained === sibling, "Streaming one lane must not remount its completed sibling")
      #expect(retained?.selectedRange() == selected)
      #expect(abs((horizontal?.contentView.bounds.origin.x ?? 0) - 120) < 1)
    }
  }

  @Test func laneHeadersRenderAtNarrowAndWideWidthsInBothThemes() throws {
    _ = NSApplication.shared
    let transcript = try fixture(count: 4)
    for dark in [false, true] {
      for width: CGFloat in [212, 480] {
        for message in transcript.blocks[1].messages {
          let view = NativeCompareHeader(message: message)
            .frame(width: width).environment(\.colorScheme, dark ? .dark : .light)
          let host = NSHostingController(rootView: view)
          let size = host.sizeThatFits(in: CGSize(width: width, height: 300))
          #expect(size.width <= width + 1)
          #expect(size.height >= 20 && size.height <= 80)
          // Render directly, without presenting a test window or enabling
          // system accessibility. Compare text selection is tested separately.
          let renderer = ImageRenderer(content: view)
          renderer.scale = 2
          let image = try #require(renderer.nsImage)
          let data = try #require(image.tiffRepresentation)
          let pixels = try #require(NSBitmapImageRep(data: data))
          #expect(abs(CGFloat(pixels.pixelsWide) / 2 - width) <= 1)
          var ink = 0
          // Exclude the bottom divider; text/mark coverage does not scale
          // with lane width (a wide lane still has the same short title).
          for y in 0..<max(0, pixels.pixelsHigh - 4) {
            for x in 0..<pixels.pixelsWide {
              if (pixels.colorAt(x: x, y: y)?.alphaComponent ?? 0) > 0.1 { ink += 1 }
            }
          }
          #expect(ink > 100, "Header cannot be a blank surface or just its divider")
          if let directory = ProcessInfo.processInfo.environment["PADDOCK_COMPARE_CAPTURE_DIR"] {
            let file = URL(fileURLWithPath: directory).appending(
              path: "header-\(message.id)-\(Int(width))-\(dark ? "dark" : "light").png")
            try pixels.representation(using: .png, properties: [:])?.write(to: file)
          }
          #expect(!descendants(host.view).contains { $0 is WKWebView })
        }
      }
    }
  }

  private func descendants(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(descendants)
  }
  private func fixture(count: Int, delta: String = "") throws -> StudioState.NativeTranscript {
    let question: [String: Any] = [
      "id": "question", "role": "user", "text": "Compare these", "model": "", "reasoning": "",
      "streaming": false, "stopped": false, "error": "", "incomplete": false,
    ]
    let lanes: [[String: Any]] = (0..<count).map { i in
      // Outside the literal: inline, Swift 6.3.3 cannot type-check it in time.
      let text: String =
        "## Lane \(i)\n\nSelectable **Markdown**.\n\n```swift\nlet x = \(i)\n```"
        + (i == 0 ? delta : "")
      return [
        "id": "lane-\(i)", "group": "lane-0", "role": "assistant",
        "text": text, "model": i == 0 ? "qwen" : "cloud:account:meta/muse@meta",
        "reasoning": "", "streaming": i == 0, "stopped": false, "error": "", "incomplete": false,
        "contended": i == 0,
        "chrome": [
          "modelName": ["Qwen", "Muse", "Grok 4.2", "Kimi K3 (deepinfra/bf16)"][i],
          "vendor": ["Alibaba", "Meta", "xAI", "Moonshot"][i], "spec": "",
          "tools": ["artifacts"], "fastest": false,
          "footer": "120 tokens · 60 tok/s · 5.6s", "footerHint": "Fixture",
          "thinkingLabel": "Thinking", "thinkingMeta": "", "cutNote": "", "sections": [],
          "promptText": "",
        ],
      ]
    }
    return try JSONDecoder().decode(
      StudioState.NativeTranscript.self,
      from: JSONSerialization.data(withJSONObject: [
        "available": true, "notice": "", "conversationId": "fixture", "leafId": "lane-0",
        "messages": [question] + lanes,
      ]))
  }
}
