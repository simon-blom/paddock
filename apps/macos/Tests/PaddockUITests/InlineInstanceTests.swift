import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Inline instance creation", .serialized) @MainActor
struct InlineInstanceTests {
  @Test func configurationSharesTheInstancesScrollAndNeverStartsOnMount() async throws {
    _ = NSApplication.shared
    let client = InlineInstanceClient()
    let workspace = WorkspaceModel(client: client)
    let snapshot = try endpointFixture()
    for dark in [false, true] {
      let creation = StartModelView(
        snapshot: snapshot, model: "qwen3.8-27b", artifact: "mlx-4bit",
        client: client, onBrowse: {}, onClose: {},
        onSubmit: { _ in
          Issue.record("Opening configuration cannot start an instance")
          return false
        })
      let host = NSHostingController(
        rootView: EndpointsView(
          workspace: workspace,
          snapshot: snapshot, onCreate: {}, creation: AnyView(creation)
        )
        .environment(\.colorScheme, dark ? .dark : .light))
      host.sizingOptions = []
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 640, height: 740),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentViewController = host
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      for width: CGFloat in [640, 1000, 640] {
        window.setContentSize(NSSize(width: width, height: 740))
        try await Task.sleep(for: .milliseconds(100))
        host.view.layoutSubtreeIfNeeded()
        let scrolls = Self.views(host.view).compactMap { $0 as? NSScrollView }
        #expect(scrolls.count == 1, "No nested setup page or separately scrolling form")
        let scroll = try #require(scrolls.first)
        #expect(abs(scroll.bounds.height - 740) < 1)
        #expect(scroll.documentView?.frame.height ?? 0 > 740, "All fields remain in the page")
        #expect(host.sizeThatFits(in: NSSize(width: width, height: 740)).width <= width)
        #expect(EndpointRow.rows(snapshot: snapshot, latestJob: nil).map(\.port) == [12345])
      }
    }
    #expect(await client.preparations == 2, "Resize must not recreate the settings draft")
    #expect(await client.mutations == 0)
    await workspace.shutdown()
  }

  private static func views(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(views) }
}

private actor InlineInstanceClient: ManagerLoading {
  var preparations = 0
  var mutations = 0
  func snapshot() async throws -> ManagerSnapshot { try endpointFixture() }
  func prepareEndpoint(model: String, artifact: String) async throws -> ConfiguredEndpoint {
    preparations += 1
    return try ManagerWire.decode(
      ConfiguredEndpoint.self,
      from: Data(
        #"{"port":0,"running":false,"model":"qwen3.8-27b","artifact":"mlx-4bit","settings":{"host":"127.0.0.1","max_ctx":4096,"max_batch":1,"spec":"off","kv_cache_dtype":"auto","has_api_key":false,"vision":false,"forensics":false,"device":"metal"}}"#
          .utf8))
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    mutations += 1
    throw CancellationError()
  }
}
