import AppKit
import SwiftUI
import Testing
import WebKit

@testable import PaddockUI

@Suite("Minimal scrollbar integration", .serialized) @MainActor
struct MinimalScrollIntegrationTests {
  @Test func transcriptColumnRemainsCenteredWithAndWithoutOverflow() async throws {
    for height: CGFloat in [100, 2600] {
      let probe = NSView()
      let host = NSHostingController(
        rootView: PaddockScrollView(centersContent: true) {
          Color.clear.frame(width: 200, height: height).background(ScrollFrameProbe(view: probe))
        })
      let window = makeWindow(host, size: NSSize(width: 480, height: 320))
      defer { window.close() }
      try await settle(host.view)
      let scroll = try #require(scrolls(host.view).first)
      #expect(scroll.verticalScroller is PaddockScroller)
      #expect(
        abs(probe.convert(probe.bounds, to: host.view).midX - 240) < 1,
        "The scroll indicator must not shift the transcript away from the composer")
      #expect(scroll.contentView.bounds.height == 320)
      #expect(scroll.verticalScroller?.isHidden == (height <= 320))
    }
  }

  @Test func horizontalAndEditorScrollersUseTheSameNativeControl() async throws {
    let host = NSHostingController(
      rootView: VStack {
        PaddockScrollView(.horizontal) { Color.clear.frame(width: 2000, height: 60) }.frame(
          height: 100)
        PaddockTextEditor(text: .constant(String(repeating: "Editable text\n", count: 80)))
      })
    let window = makeWindow(host, size: NSSize(width: 480, height: 320))
    defer { window.close() }
    try await settle(host.view)
    let views = scrolls(host.view)
    #expect(views.count == 2)
    let horizontal = try #require(views.first { $0.hasHorizontalScroller })
    #expect(horizontal.horizontalScroller is PaddockScroller)
    let editor = try #require(views.first { $0.documentView is NSTextView })
    #expect(editor.verticalScroller is PaddockScroller)
    let text = try #require(editor.documentView as? NSTextView)
    text.setSelectedRange(NSRange(location: 4, length: 8))
    let originalScroller = editor.verticalScroller
    for _ in 0..<3 { PaddockScrollbars.install(on: editor) }
    #expect(editor.verticalScroller === originalScroller)
    #expect(text.selectedRange() == NSRange(location: 4, length: 8))
    #expect(text.isEditable)
  }

  @Test func viewerCSSIsThinInBothSchemesAndDoesNotStyleStandaloneWeb() async throws {
    var repo = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { repo.deleteLastPathComponent() }
    let css = try String(
      contentsOf: repo.appending(path: "studio/native-workspace/scrollbars.css"), encoding: .utf8)
    let config = WKWebViewConfiguration()
    config.websiteDataStore = .nonPersistent()
    let web = WKWebView(frame: NSRect(x: 0, y: 0, width: 400, height: 300), configuration: config)
    defer { web.stopLoading() }
    web.loadHTMLString(
      "<style>\(css)</style><div id='fixture' style='overflow:auto;width:200px;height:100px'><div style='width:900px;height:900px'>Content</div></div>",
      baseURL: nil)
    for _ in 0..<100 {
      if !web.isLoading,
        (try? await web.evaluateJavaScript("!!document.getElementById('fixture')")) as? Bool == true
      {
        break
      }
      try await Task.sleep(for: .milliseconds(20))
    }
    let result =
      try await web.evaluateJavaScript(
        """
        (() => {
          const root = document.documentElement, el = document.getElementById('fixture');
          const read = () => {
            const bar = getComputedStyle(el, '::-webkit-scrollbar'), thumb = getComputedStyle(el, '::-webkit-scrollbar-thumb');
            return {width:bar.width, thumb:thumb.backgroundColor, border:thumb.borderLeftWidth,
              track:getComputedStyle(el, '::-webkit-scrollbar-track').backgroundColor};
          };
          const standalone = read(); root.dataset.nativeTheme = ''; root.dataset.theme = 'light';
          const light = read(); root.dataset.theme = 'dark'; return {standalone, light, dark:read()};
        })()
        """) as? [String: [String: String]]
    let styles = try #require(result)
    #expect(styles["standalone"]?["width"] != "10px")
    for scheme in ["light", "dark"] {
      #expect(styles[scheme]?["width"] == "10px")
      #expect(styles[scheme]?["border"] == "3px")
      #expect(styles[scheme]?["track"] == "rgba(0, 0, 0, 0)")
    }
    #expect(styles["light"]?["thumb"] == "rgba(0, 0, 0, 0.28)")
    #expect(styles["dark"]?["thumb"] == "rgba(255, 255, 255, 0.28)")
  }

  @Test func settingsFormUsesTheSharedScrollbarToo() async throws {
    let host = NSHostingController(
      rootView: DesktopSettingsSurface {
        Section("Fixture") { ForEach(0..<40) { Text("Setting \($0)") } }
      })
    let window = makeWindow(host, size: NSSize(width: 530, height: 580))
    defer { window.close() }
    try await settle(host.view)
    let views = scrolls(host.view)
    #expect(!views.isEmpty)
    #expect(
      views.filter(\.hasVerticalScroller).allSatisfy { $0.verticalScroller is PaddockScroller })
  }

  private func makeWindow<Content: View>(_ host: NSHostingController<Content>, size: NSSize)
    -> NSWindow
  {
    _ = NSApplication.shared
    let window = NSWindow(
      contentRect: NSRect(origin: NSPoint(x: -12000, y: -12000), size: size),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    host.sizingOptions = []
    window.contentViewController = host
    window.setContentSize(size)
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    return window
  }
  private func settle(_ view: NSView) async throws {
    for _ in 0..<6 {
      view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(30))
    }
  }
  private func scrolls(_ root: NSView) -> [NSScrollView] {
    ((root as? NSScrollView).map { [$0] } ?? []) + root.subviews.flatMap(scrolls)
  }
}

private struct ScrollFrameProbe: NSViewRepresentable {
  let view: NSView
  func makeNSView(context: Context) -> NSView { view }
  func updateNSView(_ nsView: NSView, context: Context) {}
}
