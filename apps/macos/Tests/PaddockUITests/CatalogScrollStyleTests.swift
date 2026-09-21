import AppKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Minimal shared scrollbar", .serialized) @MainActor
struct CatalogScrollStyleTests {
  @Test func overflowingCatalogPaintsAThinThumbEvenWhileIdle() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      let host = NSHostingController(
        rootView:
          ScrollView {
            Color.clear.frame(height: 2600).background(PaddockScrollStyle())
          }.background(dark ? Color.black : Color.white)
          .environment(\.colorScheme, dark ? .dark : .light))
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 320, height: 600),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      host.sizingOptions = []
      window.contentViewController = host
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      window.setContentSize(NSSize(width: 320, height: 600))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for _ in 0..<4 {
        host.view.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(40))
      }
      let scroll = try #require(scrolls(host.view).first)
      for offset: CGFloat in [0, 1400] {
        scroll.contentView.scroll(to: NSPoint(x: 0, y: offset))
        scroll.reflectScrolledClipView(scroll.contentView)
        // Longer than an overlay fade; inspect actual composited pixels, not
        // the scroller's geometry or a direct call to its drawKnob override.
        try await Task.sleep(for: .milliseconds(1800))
        host.view.layoutSubtreeIfNeeded()
        let bitmap = try #require(host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds))
        host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
        let scale = CGFloat(bitmap.pixelsWide) / host.view.bounds.width
        var paintedColumns: Set<Int> = []
        for x in (bitmap.pixelsWide - Int(20 * scale))..<bitmap.pixelsWide {
          for y in 0..<bitmap.pixelsHigh {
            guard let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB) else {
              continue
            }
            let luminance = (color.redComponent + color.greenComponent + color.blueComponent) / 3
            if dark ? luminance > 0.08 : luminance < 0.92 { paintedColumns.insert(x) }
          }
        }
        #expect(paintedColumns.count >= Int(3 * scale), "Idle scrollbar disappeared")
        #expect(paintedColumns.count <= Int(5 * scale), "Scrollbar widened or painted a track")
      }
    }
  }

  private func scrolls(_ view: NSView) -> [NSScrollView] {
    (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap(scrolls)
  }

  @Test func thumbStaysThinWhileNativeHitAreaExpands() {
    for width: CGFloat in [10, 12, 15, 18] {
      let slot = NSRect(x: 0, y: 100, width: width, height: 60)
      let thumb = PaddockScroller.thumb(in: slot)
      #expect(thumb.width == 4)
      #expect(slot.contains(thumb))
      #expect(thumb.height == 60)
      #expect(slot.midX == thumb.midX)
      let horizontal = PaddockScroller.thumb(
        in: NSRect(x: 100, y: 0, width: 60, height: width), horizontal: true)
      #expect(horizontal.height == 4 && horizontal.width == 60)
      #expect(horizontal.midY == width / 2)
    }
    #expect(!PaddockScroller.isCompatibleWithOverlayScrollers)
  }

  @Test func installerIsScopedAndKeepsNativeScrollGeometry() async throws {
    _ = NSApplication.shared
    let host = NSHostingController(
      rootView:
        HStack {
          ScrollView {
            Color.clear.frame(height: 2600).background(PaddockScrollStyle())
          }
          ScrollView { Color.clear.frame(height: 2600) }
        })
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 640, height: 600),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    host.sizingOptions = []
    window.contentViewController = host
    window.setContentSize(NSSize(width: 640, height: 600))
    defer { window.close() }
    for _ in 0..<4 {
      host.view.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(30))
    }
    func scrolls(_ view: NSView) -> [NSScrollView] {
      (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap(scrolls)
    }
    let views = scrolls(host.view)
    #expect(views.count == 2)
    let catalog = try #require(views.first { $0.verticalScroller is PaddockScroller })
    #expect(views.filter { $0.verticalScroller is PaddockScroller }.count == 1)
    #expect(catalog.scrollerStyle == .legacy)
    let scroller = try #require(catalog.verticalScroller)
    for offset: CGFloat in [0, 700, 2000, 400, 0] {
      catalog.contentView.scroll(to: NSPoint(x: 0, y: offset))
      catalog.reflectScrolledClipView(catalog.contentView)
      host.view.layoutSubtreeIfNeeded()
      #expect(abs(catalog.documentVisibleRect.minY - offset) < 1)
      #expect(abs(scroller.knobProportion - 600 / 2600) < 0.001)
      #expect(catalog.verticalScroller === scroller)
      #expect(scroller.target != nil && scroller.action != nil)
    }
  }
}
