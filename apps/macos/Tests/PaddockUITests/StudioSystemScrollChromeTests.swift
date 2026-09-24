import AppKit
import PaddockNativeMarkdown
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("System conversation scroll chrome", .serialized) @MainActor
struct StudioSystemScrollChromeTests {
  @Test func headerSitsOutsideViewportWhileComposerRemainsInside() async throws {
    guard #available(macOS 26.0, *) else { return }
    _ = NSApplication.shared
    let host = NSHostingController(
      rootView: ConversationSelectionSurface(items: []) {
        ScrollView {
          VStack { ForEach(0..<100) { Text("A native conversation line \($0)") } }
            .frame(maxWidth: .infinity)
        }
        .modifier(
          StudioConversationBars {
            Color.clear.frame(height: 40)
          } footer: {
            TextField("Question", text: .constant("Draft"))
              .padding(20).frame(height: 140).background(.background)
          }
        )
        .frame(maxWidth: .infinity, maxHeight: .infinity)
      }.frame(width: 800, height: 700))
    host.sizingOptions = []
    host.safeAreaRegions = []
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 800, height: 700),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setContentSize(NSSize(width: 800, height: 700))
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for _ in 0..<10 {
      host.view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(30))
    }
    func find<T: NSView>(_ type: T.Type, in view: NSView) -> T? {
      (view as? T) ?? view.subviews.lazy.compactMap { find(type, in: $0) }.first
    }
    let scroll = try #require(find(NSScrollView.self, in: host.view))
    #expect(scroll.bounds.width == 800)
    #expect(scroll.bounds.height == 660)
    #expect(scroll.contentInsets.top == 0 && scroll.contentInsets.bottom == 140)
  }
}
