import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("One app, unified Settings", .serialized) @MainActor
struct UnifiedSettingsTests {
  @Test func everySettingsEntryPreservesTheConversationAndDraft() async {
    let workspace = WorkspaceModel(client: SelectionProbe())
    workspace.navigation.studio = .chats
    workspace.navigation.sidebarVisible = false
    workspace.draft.message = "Do not discard this question"
    workspace.navigation.manager = .downloads
    await workspace.handleDesktopRequest(.init(.settings))
    #expect(workspace.navigation.mode == .manager && workspace.navigation.manager == .downloads)
    #expect(workspace.navigation.sidebarVisible)
    workspace.navigation.returnToChat()
    #expect(workspace.navigation.mode == .studio && workspace.navigation.studio == .chats)
    #expect(!workspace.navigation.sidebarVisible)
    #expect(workspace.draft.message == "Do not discard this question")
    #expect(
      Set(ManagerDestination.allCases.map(\.rawValue)).isSuperset(of: [
        "Conversation", "Application", "Instances", "Catalog", "Downloads", "Cloud providers",
        "Custom endpoints", "Connectors", "This Mac",
      ]))
  }

  @Test func catalogDoesNotPrepareOrStartAModelOnMount() async throws {
    _ = NSApplication.shared
    let client = SelectionProbe()
    let snapshot = try endpointFixture()
    let workspace = WorkspaceModel(client: client)
    for action in [DesktopAction.startModel, .startSpeechModel] {
      await workspace.handleDesktopRequest(.init(action))
      #expect(workspace.navigation.manager == .models)
      #expect(workspace.navigation.libraryPurpose == (action == .startSpeechModel ? .speech : .all))
    }
    let view = ModelLibraryView(
      snapshot: snapshot, canStart: true,
      onDownload: { _, _ in Issue.record("Browsing cannot download") },
      onStart: { _, _ in Issue.record("Browsing cannot configure or start") })
    let host = NSHostingView(rootView: view)
    host.frame = NSRect(x: 0, y: 0, width: 680, height: 740)
    host.layoutSubtreeIfNeeded()
    try await Task.sleep(for: .milliseconds(60))
    #expect(await client.preparations == 0)
    #expect(await client.mutations == 0)
    await workspace.shutdown()
  }

  @Test func allCompatibleVersionsAreOfferedIncludingDownloadsButSpeechNeverPicksChat() throws {
    let base = try endpointFixture()
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"download","display":"Not installed","vendor":"IBM","capability":["chat"],"installed":false,"total_size":1,"artifacts":[{"id":"mlx","kind":"weights","format":"safetensors","label":"MLX","installed":false,"total_size":1,"backend_supported":true}]},{"id":"cuda","display":"CUDA","capability":["chat"],"installed":true,"total_size":1,"artifacts":[{"id":"cuda","kind":"weights","format":"safetensors","label":"CUDA","installed":true,"total_size":1,"backend_supported":false}]}]}"#
          .utf8))
    let snapshot = ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: catalog, runners: [])
    #expect(LibraryCatalog.entries(catalog: catalog, backend: "metal").map(\.id) == ["download"])
    #expect(LibraryCatalog.entries(catalog: catalog, backend: "metal", purpose: .speech).isEmpty)
    let view = StartModelView(snapshot: snapshot, model: "download") { _ in false }
    #expect(view.initialModel == "download" && view.request == nil)
    #expect(view.validation != nil, "A missing checkpoint cannot start before download")
  }

  @Test func catalogAndProviderMarksFitBothAppearancesWithoutVisibleTestWindows() throws {
    _ = NSApplication.shared
    let snapshot = try endpointFixture()
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: ModelLibraryView(snapshot: snapshot)
          .environment(\.colorScheme, dark ? .dark : .light))
      #expect(host.sizeThatFits(in: NSSize(width: 620, height: 740)).width <= 620)
      #expect(ProviderArtwork.image(for: "Alibaba") != nil)
    }
  }
}

private actor SelectionProbe: ManagerLoading {
  var preparations = 0
  var mutations = 0
  func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
  func prepareEndpoint(model: String, artifact: String) async throws -> ConfiguredEndpoint {
    preparations += 1
    throw CancellationError()
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    mutations += 1
    throw CancellationError()
  }
}
