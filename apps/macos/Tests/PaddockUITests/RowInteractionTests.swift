import AppKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Catalog row pointer interaction", .serialized, .timeLimit(.minutes(1))) @MainActor
struct RowInteractionTests {
  @Test(arguments: [74.0, 76.0])
  func unselectedRowRespondsToMouseInPaddingAndEmptySpace(height: Double) async throws {
    var clicks = 0
    let controller = NSHostingController(
      rootView:
        ModelListRow(height: height, selected: false, action: { clicks += 1 }) {
          HStack(spacing: 10) {
            Image(systemName: "square.stack").frame(width: 30, height: 30)
            Text("Model").font(.system(size: 12))
            Spacer()
          }
        }.frame(width: 300, height: height).background(PaddockStyle.canvas))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 300, height: height),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = controller
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(80))
    controller.view.layoutSubtreeIfNeeded()
    // Real mouse events sent directly to our test window, not AXPress (which
    // bypasses hit testing), and no global cursor movement or user-window click.
    for point in [
      NSPoint(x: 5, y: 5), NSPoint(x: 290, y: height / 2), NSPoint(x: 150, y: height - 5),
    ] {
      let before = clicks
      for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
        let event = try #require(
          NSEvent.mouseEvent(
            with: type, location: point,
            modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
            windowNumber: window.windowNumber, context: nil, eventNumber: 1,
            clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
        window.sendEvent(event)
      }
      try await Task.sleep(for: .milliseconds(30))
      #expect(clicks == before + 1, "Mouse click at \(point) did not activate the full row")
    }
  }

  @Test func cloudBrowserMouseClickChangesTheSelectedModel() async throws {
    let model = CloudBrowserModel(client: CloudFixture())
    await model.refresh()
    let controller = NSHostingController(
      rootView: CloudModelsView(model: model).frame(width: 640, height: 820))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 640, height: 820),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = controller
    window.orderBack(nil)
    defer {
      window.close()
      model.stop()
    }
    try await Task.sleep(for: .milliseconds(100))
    controller.view.layoutSubtreeIfNeeded()
    func firstScrollView(_ view: NSView) -> NSScrollView? {
      if let scroll = view as? NSScrollView { return scroll }
      return view.subviews.lazy.compactMap(firstScrollView).first
    }
    let scroll = try #require(firstScrollView(controller.view))
    let document = try #require(scroll.documentView)
    // Second row, empty space inside the selection button. The trailing
    // sibling is now an Add button; clicking it must not select the row.
    let yFromTop = 76.0 + 37
    let point = document.convert(
      NSPoint(
        x: document.bounds.maxX - 64,
        y: document.isFlipped ? yFromTop : document.bounds.maxY - yFromTop), to: nil)
    #expect(scroll.contentView.bounds.height > 200)
    #expect(controller.view.bounds.width == 640)
    #expect(controller.view.bounds.height == 820)
    let order =
      CloudOrder(rawValue: UserDefaults.standard.string(forKey: "openRouterSortOrder") ?? "")
      ?? .newest
    let rows = CloudCatalogPresentation.entries(model.models, query: "", filters: [], order: order)
    #expect(model.providerModel == rows[0].id)
    for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
      let event = try #require(
        NSEvent.mouseEvent(
          with: type, location: point,
          modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
          windowNumber: window.windowNumber, context: nil, eventNumber: 1,
          clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
      window.sendEvent(event)
    }
    try await Task.sleep(for: .milliseconds(80))
    #expect(model.providerModel == rows[1].id)
  }
}
