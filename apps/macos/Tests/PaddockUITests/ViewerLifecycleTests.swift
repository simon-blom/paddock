import AppKit
import Foundation
import PaddockClient
import Testing
import WebKit

@testable import PaddockConversationCore
@testable import PaddockStudio
@testable import PaddockUI

@Suite("Independent viewer lifetime", .serialized) @MainActor
struct ViewerLifecycleTests {
  @Test func separateWebViewsCloseIndependentlyAndKeepAnActiveGraphBridge() async throws {
    _ = NSApplication.shared
    let workspace = StudioWorkspace(client: NoCore())
    let runtime = try runtime()
    workspace.apply(try await state(runtime, document: true, graph: true))
    let document = workspace.webView(for: .document)
    let graph = workspace.webView(for: .graph)
    #expect(document !== graph)
    #expect(document.configuration.websiteDataStore !== graph.configuration.websiteDataStore)
    #expect(!document.configuration.websiteDataStore.isPersistent)
    #expect(!graph.configuration.websiteDataStore.isPersistent)
    workspace.apply(try await state(runtime, document: false, graph: true))
    #expect(!workspace.hasWebViewer(for: .document))
    #expect(workspace.hasWebViewer(for: .graph))
    #expect(workspace.webView(for: .graph) === graph)
    #expect(document.navigationDelegate == nil && graph.navigationDelegate != nil)
    workspace.apply(try await state(runtime, document: true, graph: false, busy: true))
    let reopened = workspace.webView(for: .document)
    #expect(reopened !== document)
    #expect(workspace.webView(for: .graph) === graph)
    workspace.apply(try await state(runtime, document: true, graph: false))
    #expect(!workspace.hasWebViewer(for: .graph))
    #expect(workspace.webView(for: .document) === reopened)
    #expect(graph.navigationDelegate == nil)
    await workspace.shutdown()
    await runtime.close()
    #expect(!workspace.hasWebViewer)
  }

  @Test func firstAdmissionRetainsPreparedGraphAndNavigationReleasesBothViewers() async throws {
    let workspace = StudioWorkspace(client: NoCore())
    let runtime = try runtime()
    workspace.apply(try await state(runtime, document: false, graph: true, busy: true, draft: true))
    let graph = workspace.webView(for: .graph)
    workspace.apply(try await state(runtime, document: true, graph: true, busy: true))
    #expect(workspace.webView(for: .graph) === graph)
    let document = workspace.webView(for: .document)
    workspace.apply(try await state(runtime, document: false, graph: false, conversation: "other"))
    #expect(!workspace.hasWebViewer)
    #expect(document.navigationDelegate == nil && graph.navigationDelegate == nil)
    await workspace.shutdown()
    await runtime.close()
  }

  @Test func simultaneousSlotsKeepTheirOwnWebViewWhileResizingOffscreen() async throws {
    _ = NSApplication.shared
    let workspace = StudioWorkspace(client: NoCore())
    let left = WorkspaceWebContainer()
    let right = WorkspaceWebContainer()
    let root = NSView(frame: NSRect(x: 0, y: 0, width: 800, height: 480))
    root.addSubview(left)
    root.addSubview(right)
    let window = NSWindow(
      contentRect: root.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = root
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    defer { window.close() }
    let document = workspace.webView(for: .document)
    let graph = workspace.webView(for: .graph)
    for width: CGFloat in [260, 420, 300] {
      left.frame = NSRect(x: 0, y: 0, width: width, height: 480)
      right.frame = NSRect(x: width, y: 0, width: 800 - width, height: 480)
      left.embed(document)
      right.embed(graph)
      root.layoutSubtreeIfNeeded()
      #expect(document.superview === left && graph.superview === right)
      #expect(document.frame == left.bounds && graph.frame == right.bounds)
    }
    await workspace.shutdown()
    #expect(left.subviews.isEmpty && right.subviews.isEmpty)
  }

  private func runtime() throws -> NativeStudioRuntime {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    return NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in }
  }
  private func state(
    _ runtime: NativeStudioRuntime, document: Bool, graph: Bool,
    busy: Bool = false, draft: Bool = false, conversation: String = "chat"
  ) async throws -> StudioState {
    let fields = try await runtime.viewerLifetimeFixture(
      document: document, graph: graph, busy: busy, draft: draft, conversation: conversation)
    return try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields))
  }
  private struct NoCore: ManagerLoading {
    func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
  }
}

extension NativeStudioRuntime {
  fileprivate func viewerLifetimeFixture(
    document: Bool, graph: Bool, busy: Bool, draft: Bool, conversation: String
  ) throws -> O {
    self.document = try .init(fields: ["id": .string(conversation), "messages": .array([])])
    previewPart =
      document
      ? [
        "type": .string("file"), "mime": .string("application/pdf"), "attachmentId": .string("pdf"),
        "name": .string("Paper.pdf"),
      ] : nil
    self.draft = draft
    mutating = busy
    graphVisible = graph
    revision += 1
    return presentation()
  }
}
