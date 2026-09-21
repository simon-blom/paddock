import AppKit
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Quick overlay", .serialized, .timeLimit(.minutes(1))) @MainActor
struct QuickPanelTests {
  @Test func anchorsUnderStatusItemAndGrowsDown() {
    let display = NSRect(x: 0, y: 25, width: 1512, height: 920)
    let anchor = NSRect(x: 1020, y: 945, width: 24, height: 37)
    let small = QuickPanelLayout.frame(height: 132, visible: display, anchor: anchor)
    let large = QuickPanelLayout.frame(height: 350, visible: display, anchor: anchor)
    #expect(small.width == 460)
    #expect(small.midX == anchor.midX)
    #expect(small.maxY == anchor.minY - 8)
    #expect(large.maxY == small.maxY)
    #expect(large.minY < small.minY)
  }

  @Test func clampsToBothEdgesAndNegativeOriginDisplays() {
    let display = NSRect(x: -1920, y: 25, width: 1920, height: 1000)
    for x: CGFloat in [-1920, -5] {
      let frame = QuickPanelLayout.frame(
        height: 132, visible: display,
        anchor: NSRect(x: x, y: 1025, width: 24, height: 25))
      #expect(frame.minX >= display.minX + 12)
      #expect(frame.maxX <= display.maxX - 12)
      #expect(frame.maxY == display.maxY - 8)
    }
  }

  @Test func keyboardEntryIsTopCenteredAndHeightIsBounded() {
    let display = NSRect(x: 50, y: 30, width: 600, height: 400)
    let small = QuickPanelLayout.frame(height: 1, visible: display, anchor: nil)
    #expect(small.midX == display.midX)
    #expect(small.height == 132)
    #expect(small.maxY == display.maxY - 8)
    let large = QuickPanelLayout.frame(height: 9999, visible: display, anchor: nil)
    #expect(display.contains(large))
    #expect(large.minY >= display.minY + 12)
  }

  @Test func respectsReducedMotion() {
    for showing in [true, false] {
      #expect(QuickPanelLayout.duration(showing: showing, reduceMotion: true) == 0)
      #expect(QuickPanelLayout.duration(showing: showing, reduceMotion: false) <= 0.2)
    }
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OS_UI_TEST"] == "1"))
  func staleAnimationCannotDismissReopenedDraft() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: OverlayNoCore())
    let controller = QuickQuestionController()
    controller.reduceMotion = { false }
    controller.draft.text = "Synthetic retained draft"
    controller.show(workspace: model) { _ in }
    let panel = try #require(controller.panel)
    controller.hide(restoreFocus: false)
    controller.show(workspace: model) { _ in }
    try await Task.sleep(for: .milliseconds(350))
    #expect(controller.presented)
    #expect(panel.isVisible)
    #expect(panel.alphaValue == 1)
    #expect(panel.firstResponder is DraftTextView)
    #expect(controller.draft.text == "Synthetic retained draft")
    controller.hide(restoreFocus: false)
    try await Task.sleep(for: .milliseconds(200))
    #expect(!panel.isVisible)
    await model.shutdown()
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OS_UI_TEST"] == "1"))
  func ownedControlsStayOpenButOutsideClickDismissesAndKeepsDraft() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: OverlayNoCore())
    let controller = QuickQuestionController()
    controller.reduceMotion = { true }
    controller.draft.text = "Retain me"
    controller.show(workspace: model) { _ in }
    let panel = try #require(controller.panel)
    controller.clicked(window: panel)
    #expect(controller.presented)
    let popup = NSWindow(
      contentRect: .zero, styleMask: .borderless, backing: .buffered, defer: false)
    popup.isReleasedWhenClosed = false
    defer { popup.close() }
    popup.level = .popUpMenu
    controller.clicked(window: popup)
    #expect(controller.presented)
    popup.level = .normal
    controller.clicked(window: popup)
    #expect(!controller.presented)
    #expect(!panel.isVisible)
    #expect(controller.draft.text == "Retain me")
    await model.shutdown()
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OS_UI_TEST"] == "1"))
  func statusItemOwnershipIsIdempotent() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: OverlayNoCore())
    let status = DesktopStatusController(
      workspace: model, open: { _ in }, question: { _ in }, settings: {})
    status.setVisible(true)
    let item = try #require(status.item)
    status.setVisible(true)
    #expect(status.item === item)
    #expect(item.menu != nil)
    #expect(status.anchor != nil)
    status.setVisible(false)
    #expect(status.item == nil)
    #expect(status.anchor == nil)
    await model.shutdown()
  }
}

private struct OverlayNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    throw ManagerError.core("Synthetic offline manager")
  }
}
