import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native menu and conversation chrome", .serialized) @MainActor
struct NativeChromePolishTests {
  private func row(_ port: UInt16, title: String = "Qwen 3.8 27B", status: String = "ok") throws
    -> EndpointRow
  {
    let runner = try ManagerWire.decode(
      RunnerInfo.self,
      from: JSONSerialization.data(withJSONObject: [
        "port": port, "pid": 42, "status": status, "display": title,
        "endpoint": "http://127.0.0.1:\(port)",
      ]))
    return EndpointRow(port: port, runner: runner, configured: nil, job: nil)
  }

  @Test func menuShowsCleanNamesAndKeepsDuplicateInstancesDistinguishable() throws {
    let first = try row(11543)
    #expect(first.desktopMenuTitle(in: [first]) == "Qwen 3.8 27B")
    let other = try row(11544, title: "Bonsai 2 27B")
    #expect(first.desktopMenuTitle(in: [first, other]) == "Qwen 3.8 27B")
    let duplicate = try row(11545)
    #expect(first.desktopMenuTitle(in: [first, duplicate]) == "Qwen 3.8 27B (Port 11543)")
    #expect(duplicate.desktopMenuTitle(in: [first, duplicate]) == "Qwen 3.8 27B (Port 11545)")
    #expect(first.baseURL == "http://127.0.0.1:11543/v1")
  }

  @Test func menuStillSurfacesUnhealthyAndTransitioningRunners() throws {
    for status in ["unreachable", "draining", "starting"] {
      let endpoint = try row(11543, status: status)
      let title = endpoint.desktopMenuTitle(in: [endpoint])
      #expect(title.hasSuffix(endpoint.desktopStatus))
      #expect(!title.contains(":11543") && !title.contains(" · "))
    }
    let configured = try ManagerWire.decode(
      ConfiguredEndpoint.self,
      from: JSONSerialization.data(withJSONObject: [
        "port": 11543, "display": "Qwen 3.8 27B", "running": false,
        "local_only": true, "revision": "r1",
      ]))
    let stopped = EndpointRow(port: 11543, runner: nil, configured: configured, job: nil)
    #expect(stopped.canQuickStart && stopped.desktopMenuTitle(in: [stopped]) == "Qwen 3.8 27B")
  }

  @Test func spinnerAndActionsShareOneControlHeightInBothThemes() {
    _ = NSApplication.shared
    for dark in [false, true] {
      for (busy, status, showsActions, expectedWidth): (Bool, String?, Bool, CGFloat) in [
        (true, nil, true, 60), (false, "Failed", true, 60),
        (false, nil, true, 28), (true, nil, false, 28),
      ] {
        let host = NSHostingController(
          rootView: StudioHistoryRowControls(
            title: "A long conversation title spanning several lines", id: "fixture",
            busy: busy, status: status, showsActions: showsActions
          ) {
            Button("Rename") { Issue.record("Measuring must never execute an action") }
          }.environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: NSSize(width: 100, height: 100))
        #expect(size.width == expectedWidth && size.height == 28)
      }
    }
  }

  @Test func headerTitleUsesTheChatColumnsLeadingEdgeInsteadOfWindowCenter() throws {
    for width: CGFloat in [440, 680, 900, 1200, 1800] {
      for comparison in [false, true] {
        let column = StudioColumnLayout.resolve(
          available: width, viewport: nil, comparison: comparison)
        for controls: CGFloat in [20, 240] {
          let frame = try #require(
            StudioConversationTitleLayout.frame(
              available: width, height: 44, column: column, controlsClearance: controls))
          #expect(frame.minX == max(column.minX, controls))
          #expect(frame.maxX <= column.maxX && frame.maxX <= width - 52)
          #expect(frame.height == 44)
        }
      }
    }
    // A document pane on the left owns the controls, even with Chats folded.
    let column = StudioColumnLayout.resolve(available: 600, viewport: nil)
    let besideDocument = try #require(
      StudioConversationTitleLayout.frame(
        available: 600, height: 44, column: column, controlsClearance: 20))
    #expect(besideDocument.minX == column.minX)
  }

  @Test func titleNeverOccupiesTheControlsWhenSpaceIsInsufficient() {
    let column = StudioColumnLayout.resolve(available: 320, viewport: nil)
    #expect(
      StudioConversationTitleLayout.frame(
        available: 320, height: 44, column: column, controlsClearance: 240) == nil)
    #expect(
      StudioConversationTitleLayout.frame(
        available: .infinity, height: 44, column: column, controlsClearance: 20) == nil)
  }
}
