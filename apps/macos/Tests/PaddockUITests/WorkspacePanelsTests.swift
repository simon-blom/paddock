import AppKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Full-height native sidebar") @MainActor
struct WorkspacePanelsTests {
  @Test func cardsAndGuttersFitAtEveryMinimumAndSidebarWidth() {
    for mode in WorkspaceMode.allCases {
      var navigation = WorkspaceNavigation()
      navigation.mode = mode
      navigation.sidebarVisible = true
      for available: CGFloat in [WorkspacePanelMetrics.minimumWidth(navigation), 1200, 1800] {
        for preferred: CGFloat in [0, 220, 260, 360, 1000] {
          let width = WorkspacePanelMetrics.width(
            preferred: preferred, mode: mode, available: available)
          let content =
            available - width - WorkspacePanelMetrics.gap
          #expect(WorkspacePanelMetrics.limits(mode).contains(width))
          #expect(content >= WorkspacePanelMetrics.contentMinimum)
        }
      }
    }
  }
  @Test func narrowingWindowDoesNotOverwriteTheUsersPanelPreference() {
    var navigation = WorkspaceNavigation()
    navigation.sidebarVisible = true
    navigation.panelWidth = 360
    #expect(
      WorkspacePanelMetrics.width(preferred: navigation.panelWidth, mode: .studio, available: 920)
        == 276)
    #expect(navigation.panelWidth == 360)
    #expect(
      WorkspacePanelMetrics.width(preferred: navigation.panelWidth, mode: .studio, available: 1200)
        == 360)
    navigation.mode = .manager
    #expect(navigation.panelWidth == 220)
    navigation.panelWidth = 250
    navigation.mode = .studio
    #expect(navigation.panelWidth == 360)
    navigation.mode = .manager
    #expect(navigation.panelWidth == 250)
  }
  @Test func gripAndPanelHaveNoIntrinsicViewportInflation() {
    _ = NSApplication.shared
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: HStack(spacing: 0) {
          Color.clear.frame(width: 220).workspacePanel(PaddockStyle.sidebar)
          WorkspacePanelGrip(width: .constant(220), range: 220...360, label: "Resize chats panel")
          Color.clear.workspacePanel(PaddockStyle.canvas)
        }.environment(\.colorScheme, dark ? .dark : .light))
      let size = CGSize(width: 900, height: 600)
      host.view.frame = NSRect(origin: .zero, size: size)
      host.view.layoutSubtreeIfNeeded()
      let fit = host.sizeThatFits(in: size)
      #expect(fit.width <= size.width && fit.height <= size.height)
    }
  }
}
