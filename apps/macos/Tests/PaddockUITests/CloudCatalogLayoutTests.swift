import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Cloud and local catalog layout parity", .serialized) @MainActor
struct CloudCatalogLayoutTests {
  @Test func detailColumnUsesFullHeightAndTheSameSplitWidthAsLocalCatalog() async throws {
    _ = NSApplication.shared
    let browser = CloudBrowserModel(client: CloudFixture())
    await browser.refresh()
    let picks = [
      CloudModelPick(id: "anthropic/claude-test"),
      try JSONDecoder().decode(
        CloudModelPick.self,
        from: Data(#"{"id":"unlisted-model","provider":"pinned-provider"}"#.utf8)),
    ]
    let account = try CloudWorkflowFixture.row(models: picks)
    let client = CloudWorkflowFixture(rows: [account])
    let connections = ConnectionsModel(client: client)
    await connections.refresh()
    defer { browser.stop() }
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: CloudModelsView(model: browser, connections: connections)
          .environment(\.colorScheme, dark ? .dark : .light))
      host.sizingOptions = []
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 640, height: 650),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentViewController = host
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for width: CGFloat in [640, 1100, 640] {
        window.setContentSize(NSSize(width: width, height: 650))
        try await Task.sleep(for: .milliseconds(100))
        host.view.layoutSubtreeIfNeeded()
        let scrolls = scrollViews(host.view)
        let detail = try #require(
          scrolls.max {
            $0.convert($0.bounds, to: host.view).minX < $1.convert($1.bounds, to: host.view).minX
          })
        let frame = detail.convert(detail.bounds, to: host.view)
        let listWidth = CatalogColumns<Color, Color>.listWidth(
          preferred: 310, available: width, limits: 240...400)
        #expect(abs(frame.minX - listWidth - WorkspacePanelMetrics.gap) < 1)
        #expect(
          abs(frame.minY) < 1 && abs(frame.height - 650) < 1,
          "Provider/account/saved-model controls must not push down the detail column: \(frame)")
        #expect(abs(frame.maxX - width) < 1)
      }
    }
    #expect(await client.modelWrites == 0)
    #expect(await client.unlocks == 0)
    await connections.stop()
  }

  @Test func missingOrLockedPrivateAccountNeverBorrowsThePublicCatalog() throws {
    let locked = try CloudWorkflowFixture.row(credentialReady: false)
    #expect(CloudModelsView.canBrowse(.openrouter, account: nil))
    #expect(CloudModelsView.canBrowse(.openrouter, account: locked))
    for service in [CloudService.openai, .anthropic, .custom] {
      #expect(!CloudModelsView.canBrowse(service, account: nil))
      #expect(!CloudModelsView.canBrowse(service, account: locked))
      #expect(CloudModelsView.canBrowse(service, account: try CloudWorkflowFixture.row()))
    }
  }

  @Test func compactAccountsAndSavedPinsFitTheNarrowestList() async throws {
    let picks = [
      CloudModelPick(id: "same/model"),
      try JSONDecoder().decode(
        CloudModelPick.self,
        from: Data(
          #"{"id":"same/model","provider":"region-specific-provider-with-a-long-name"}"#.utf8)),
    ]
    let accounts = try [
      CloudWorkflowFixture.row(id: "first account with a very long name", models: picks),
      CloudWorkflowFixture.row(id: "second"),
    ]
    let connections = ConnectionsModel(client: CloudWorkflowFixture(rows: accounts))
    await connections.refresh()
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: VStack {
          CloudAccountView(
            model: connections, service: .openrouter, account: accounts[0], compact: true)
          CloudSavedPicksView(account: accounts[0], model: connections, compact: true)
        }.padding(16).environment(\.colorScheme, dark ? .dark : .light))
      let size = host.sizeThatFits(in: CGSize(width: 240, height: 400))
      #expect(size.width <= 240)
    }
    #expect(connections.rows.first?.models.map(\.pickKey) == picks.map(\.pickKey))
    await connections.stop()
  }

  private func scrollViews(_ view: NSView) -> [NSScrollView] {
    if let scroll = view as? NSScrollView { return [scroll] }
    return view.subviews.flatMap(scrollViews)
  }
}
