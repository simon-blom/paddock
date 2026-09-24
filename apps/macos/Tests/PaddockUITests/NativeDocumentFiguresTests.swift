import AppKit
import PaddockStudio
import SwiftUI
import Testing
import WebKit

@testable import PaddockUI

@Suite("Native document figure visibility", .serialized)
struct NativeDocumentFiguresTests {
  @MainActor @Observable final class Probe {
    var loads = 0
    var top: CGFloat = 800
  }
  @MainActor struct Fixture: View {
    let probe: Probe
    let page: StudioState.NativeTranscript.Message.DocumentResult.Page
    let pixels: CGImage
    let dark: Bool
    var body: some View {
      ScrollView {
        VStack(spacing: 0) {
          Color.clear.frame(height: probe.top)
          // Compare has an inner horizontal scroll view, regular chat doesn't.
          ScrollView(.horizontal) {
            NativeDocumentFigures(
              page: page,
              load: {
                probe.loads += 1
                return pixels
              }, onOpen: {}
            ).frame(width: 360)
          }.fixedSize(horizontal: false, vertical: true)
          Color.clear.frame(height: 600)
        }
      }.defaultScrollAnchor(.top)
        .environment(\.colorScheme, dark ? .dark : .light)
    }
  }
  @Test @MainActor func onlyVisibleFigurePagesLoadIncludingNestedCompareInBothThemes() async throws
  {
    let data = Data(
      #"{"id":1,"state":"done","text":"text","note":"","attachmentID":"source","regions":[{"label":"figure","text":"","boxes":[[0,0,999,999]],"quads":[]}],"unsure":[]}"#
        .utf8)
    let page = try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.DocumentResult.Page.self, from: data)
    let context = try #require(
      CGContext(
        data: nil, width: 200, height: 100, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.setFillColor(CGColor(gray: 0.5, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: 200, height: 100))
    let pixels = try #require(context.makeImage())
    _ = NSApplication.shared
    for dark in [false, true] {
      let probe = Probe()
      let host = NSHostingView(
        rootView: Fixture(probe: probe, page: page, pixels: pixels, dark: dark))
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 400, height: 400), styleMask: [.titled],
        backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentView = host
      window.orderBack(nil)
      defer { window.close() }
      for _ in 0..<20 {
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(10))
      }
      #expect(probe.loads == 0, "An offscreen compare page must not fetch its original")
      probe.top = 0
      for _ in 0..<100 {
        host.layoutSubtreeIfNeeded()
        if probe.loads > 0 { break }
        try await Task.sleep(for: .milliseconds(10))
      }
      #expect(probe.loads == 1)
      probe.top = 800
      for _ in 0..<20 {
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(10))
      }
      #expect(probe.loads == 1)
      probe.top = 0
      for _ in 0..<100 {
        host.layoutSubtreeIfNeeded()
        if probe.loads == 2 { break }
        try await Task.sleep(for: .milliseconds(10))
      }
      #expect(
        probe.loads == 2,
        "Reentering the viewport must reacquire rather than retain an offscreen bitmap")
      func hasWeb(_ view: NSView) -> Bool {
        view is WKWebView || view.subviews.contains(where: hasWeb)
      }
      #expect(!hasWeb(host))
    }
  }
}
