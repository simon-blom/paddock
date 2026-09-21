import AppKit
import Foundation
import PaddockStudio
import SwiftUI
import Testing
import WebKit

@testable import PaddockNativeMarkdown
@testable import PaddockUI

@Suite("Native rendering boundary", .serialized)
struct NativeRenderingBoundaryTests {
  @Test func framesAreAtomicUnicodeSafeAndBounded() throws {
    let original = Data(String(repeating: "🦊\n\u{0}", count: 60000).utf8)
    var assembler = StudioPresentationFrames()
    let size = 48 * 1024
    let count = (original.count + size - 1) / size
    var result: Data?
    for i in 0..<count {
      result = try assembler.accept([
        "transfer": 8, "index": i, "count": count, "bytes": original.count,
        "payload": original.subdata(in: i * size..<min((i + 1) * size, original.count))
          .base64EncodedString(),
      ])
      if i < count - 1 { #expect(result == nil) }
    }
    #expect(result == original)
    #expect(throws: (any Error).self) {
      try assembler.accept([
        "transfer": 9, "index": 1, "count": 2, "bytes": 50000, "payload": "YQ==",
      ])
    }
    #expect(throws: (any Error).self) {
      try assembler.accept([
        "transfer": 9, "index": 0, "count": 1, "bytes": Int.max, "payload": "YQ==",
      ])
    }
  }
  @Test func htmlAndImagesCannotReachLibraryWebRenderers() {
    let raw = """
      ## A 🦊 heading

      <iframe src="https://example.com"></iframe>

      <svg><circle r="10"/></svg>

      ![Remote](data:image/svg+xml;base64,abc)

      ```html
      <b>Keep the original code fence</b>
      ```
      """
    let safe = NativeMarkdownPolicy.source(raw)
    #expect(safe.contains("```html\n<iframe"))
    #expect(safe.contains("\\![Remote]"))
    #expect(safe.contains("<b>Keep the original code fence</b>"))
    #expect(safe.contains("## A 🦊 heading"))
    #expect(
      NativeMarkdownPolicy.source("Plain **Markdown**\n\n$E=mc^2$")
        == "Plain **Markdown**\n\n$E=mc^2$")
  }
  @Test @MainActor func hostileMarkdownRendersWithoutAnyWebView() async throws {
    _ = NSApplication.shared
    let text = """
      # Native

      <div><iframe src="https://example.com"></iframe></div>

      <svg><circle r="10"/></svg>

      ![Image](https://example.com/image.svg)

      $$E=mc^2$$

      ```mermaid
      flowchart LR
        A --> B
      ```
      """
    let host = NSHostingView(rootView: NativeMarkdown(text).frame(width: 740))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 780, height: 900), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    defer { window.close() }
    // Offscreen mounting only: no synthetic fixture windows on the desktop.
    try await Task.sleep(for: .milliseconds(700))
    host.layoutSubtreeIfNeeded()
    #expect(!containsWebView(host))
    #expect(host.fittingSize.height > 200)
  }
  @Test func sourceLinksRejectLocalFilesAndExecutableSchemes() {
    for url in [
      "file:///etc/passwd", "javascript:alert(1)", "data:text/html,a",
      "https://user:secret@example.com",
    ] {
      #expect(NativeSearchCallView.safeURL(url) == nil)
    }
    #expect(NativeSearchCallView.safeURL("https://example.com/source") != nil)
  }
  @Test @MainActor func toolApprovalSearchAndDocumentCardsAreNativeInBothThemes() async throws {
    let data = Data(
      #"""
      {"id":"answer","role":"assistant","text":"Text","reasoning":"","model":"fixture",
       "streaming":false,"stopped":false,"error":"","incomplete":false,
       "toolCalls":[{"id":"tool","name":"write_file","server":"files","arguments":"{\"text\":\"<script>\"}","output":"","status":"pending","error":"","approvalId":"approval"}],
       "searches":[{"id":"search","query":"Evidence","provider":"brave","status":"completed","error":"","sources":[{"title":"Source","url":"https://example.com"}]}],
       "documentResult":{"facts":[{"label":"Pages","value":"1"}],"pages":[{"id":1,"state":"review","text":"Extracted **document** text","note":"Review this page","regions":[{"label":"text","text":"Extracted","boxes":[[1,2,3,4]],"quads":[]}],"unsure":[{"label":"Extracted","value":"20%"}]}]}}
      """#.utf8)
    let message = try JSONDecoder().decode(StudioState.NativeTranscript.Message.self, from: data)
    #expect(message.toolCalls?.first?.approvalId == "approval")
    for appearance in [ColorScheme.light, .dark] {
      let host = NSHostingView(
        rootView: NativeStudioMessage(message: message).frame(width: 700).environment(
          \.colorScheme, appearance))
      let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 740, height: 900), styleMask: [.titled],
        backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentView = host
      try await Task.sleep(for: .milliseconds(200))
      host.layoutSubtreeIfNeeded()
      #expect(!containsWebView(host))
      #expect(host.fittingSize.height > 250)
      window.close()
    }
  }
  @MainActor private func containsWebView(_ view: NSView) -> Bool {
    view is WKWebView || view.subviews.contains(where: containsWebView)
  }
}
