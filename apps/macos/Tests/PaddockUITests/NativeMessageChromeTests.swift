import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native message chrome", .serialized) @MainActor
struct NativeMessageChromeTests {
  @Test func oldProjectionRemainsDecodableWithoutInventingStatistics() throws {
    let old =
      #"{"id":"old","role":"assistant","text":"Saved reply","reasoning":"","model":"old-model","streaming":false,"stopped":false,"error":"","incomplete":false}"#
    let message = try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.self, from: Data(old.utf8))
    #expect(message.chrome == nil)
    #expect(message.model == "old-model")
  }

  @Test func headerFooterAndDetailsFitNarrowAndWideInBothAppearances() async throws {
    _ = NSApplication.shared
    let message = try NativeMessageSample.message()
    let chrome = try #require(message.chrome)
    #expect(ProviderArtwork.image(for: chrome.vendor) != nil)
    #expect(chrome.sections.map(\.id) == ["provenance", "metrics", "gpu"])
    for width: CGFloat in [280, 760] {
      for dark in [false, true] {
        let controller = NSHostingController(
          rootView: NativeStudioMessage(message: message)
            .preferredColorScheme(dark ? .dark : .light))
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: width, height: 900),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        controller.sizingOptions = []
        window.contentViewController = controller
        window.setContentSize(CGSize(width: width, height: 900))
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        // Mounting Markdown is asynchronous. Wait for actual layout, rather
        // than assuming a detached view has rendered within a fixed 100 ms.
        var fitted = CGSize.zero
        for _ in 0..<100 {
          controller.view.layoutSubtreeIfNeeded()
          fitted = controller.sizeThatFits(in: CGSize(width: width, height: 10000))
          if fitted.height > 200 { break }
          try await Task.sleep(for: .milliseconds(20))
        }
        #expect(fitted.width <= width + 1)
        #expect(fitted.height > 200)
      }
    }
    let details = NSHostingController(rootView: NativeRunDetails(chrome: chrome))
    #expect(details.sizeThatFits(in: CGSize(width: 420, height: 540)).width == 420)
  }

  @Test func footerArrivalDoesNotRemountMarkdownOrClearSelection() async throws {
    _ = NSApplication.shared
    let text = "Select this first paragraph.\n\nKeep the second paragraph selected too."
    let host = NSHostingView(
      rootView: NativeStudioMessage(
        message: try NativeMessageSample.message(text: text, streaming: true)))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 760, height: 700), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    func surface(_ view: NSView) -> NSTextView? {
      if let text = view as? NSTextView, text.string.contains("Select this first") { return text }
      return view.subviews.lazy.compactMap { surface($0) }.first
    }
    for _ in 0..<100 {
      host.layoutSubtreeIfNeeded()
      if surface(host) != nil { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    let original = try #require(surface(host))
    let range = NSRange(location: 0, length: (original.string as NSString).length)
    original.setSelectedRange(range)
    host.rootView = NativeStudioMessage(message: try NativeMessageSample.message(text: text))
    try await Task.sleep(for: .milliseconds(100))
    host.layoutSubtreeIfNeeded()
    let settled = try #require(surface(host))
    #expect(settled === original)
    #expect(settled.selectedRange() == range)
  }
}
