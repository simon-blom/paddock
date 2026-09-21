import AppKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native dropdown interaction", .serialized, .timeLimit(.minutes(1))) @MainActor
struct DropdownTests {
  // NSMenu owns a nested AppKit event loop. Keep other rendering tests from
  // closing/replacing hosting windows while this menu is tracking; check.sh
  // runs this gate in its own process after the ordinary parallel test suite.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_NATIVE_MENU_TEST"] == "1"))
  func mouseOpensNativeMenuAndSelectionUpdatesBinding() async throws {
    let state = ChoiceState()
    let controller = NSHostingController(
      rootView: ChoiceFixture(state: state))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 240, height: 80),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = controller
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(80))
    controller.view.layoutSubtreeIfNeeded()
    let observer = MenuProbe()
    observer.selectItem = "MLX"
    NotificationCenter.default.addObserver(
      observer, selector: #selector(MenuProbe.didBegin(_:)),
      name: NSMenu.didBeginTrackingNotification, object: nil)
    defer { NotificationCenter.default.removeObserver(observer) }
    // Empty trailing padding, not the label or indicator. AXPress alone would
    // miss a broken pointer hit region, as the earlier row regression showed.
    for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
      let event = try #require(
        NSEvent.mouseEvent(
          with: type, location: NSPoint(x: 236, y: 40), modifierFlags: [],
          timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
          context: nil, eventNumber: 1, clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
      window.sendEvent(event)
    }
    try await Task.sleep(for: .milliseconds(120))
    #expect(observer.opened)
    #expect(observer.checked)
    #expect(observer.disabled)
    #expect(state.selection == "MLX")
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_NATIVE_MENU_TEST"] == "1"))
  func providerMenuUsesSmallImagesAndStillSelects() async throws {
    let state = ChoiceState()
    let vendors = ["OpenRouter", "OpenAI", "Anthropic", "IBM", "KBLab"]
    let controller = NSHostingController(
      rootView:
        Dropdown(title: "Provider", value: state.selection, fillsWidth: true) {
          ForEach(vendors, id: \.self) { vendor in
            Button {
              state.selection = vendor
            } label: {
              ModelProviderMenuLabel(title: vendor, vendor: vendor)
            }
          }
          Button {
            state.selection = "Custom"
          } label: {
            ModelProviderMenuLabel(title: "Custom", vendor: nil)
          }
        }.frame(width: 240, height: 80))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 240, height: 80),
      styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = controller
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(80))
    controller.view.layoutSubtreeIfNeeded()
    let observer = MenuProbe()
    observer.selectItem = "Anthropic"
    NotificationCenter.default.addObserver(
      observer, selector: #selector(MenuProbe.didBegin(_:)),
      name: NSMenu.didBeginTrackingNotification, object: nil)
    defer { NotificationCenter.default.removeObserver(observer) }
    for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
      let event = try #require(
        NSEvent.mouseEvent(
          with: type, location: NSPoint(x: 236, y: 40), modifierFlags: [],
          timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
          context: nil, eventNumber: 1, clickCount: 1, pressure: type == .leftMouseDown ? 1 : 0))
      window.sendEvent(event)
    }
    try await Task.sleep(for: .milliseconds(120))
    #expect(observer.opened)
    for vendor in vendors {
      let size = try #require(observer.imageSizes[vendor], "Missing menu logo for \(vendor)")
      #expect(size.width > 0 && size.height > 0)
      #expect(size.width <= 16 && size.height <= 16, "Oversized menu logo for \(vendor): \(size)")
    }
    #expect(observer.imageSizes["Custom"] != nil)
    #expect(state.selection == "Anthropic")
  }
}

/// AppKit posts menu tracking notifications synchronously on the main thread.
@MainActor private final class MenuProbe: NSObject {
  var opened = false
  var checked = false
  var disabled = false
  var selectItem: String?
  var imageSizes: [String: NSSize] = [:]

  @objc func didBegin(_ notification: Notification) {
    guard let menu = notification.object as? NSMenu else { return }
    opened = true
    checked = menu.items.contains { $0.title == "All formats" && $0.state == .on }
    disabled = menu.items.contains { $0.title == "Unavailable" && !$0.isEnabled }
    for item in menu.items {
      if let image = item.image { imageSizes[item.title] = image.size }
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
      if let index = menu.items.firstIndex(where: { $0.title == self.selectItem }) {
        menu.performActionForItem(at: index)
      }
      menu.cancelTrackingWithoutAnimation()
    }
  }
}

@MainActor @Observable private final class ChoiceState {
  var selection = "All formats"
}

private struct ChoiceFixture: View {
  @Bindable var state: ChoiceState
  var body: some View {
    Dropdown(title: "Format", value: state.selection, fillsWidth: true) {
      Picker("Format", selection: $state.selection) {
        Text("All formats").tag("All formats")
        Text("MLX").tag("MLX")
      }.pickerStyle(.inline)
      Toggle("Unavailable", isOn: .constant(false)).disabled(true)
    }.frame(width: 240, height: 80).background(PaddockStyle.canvas)
  }
}
