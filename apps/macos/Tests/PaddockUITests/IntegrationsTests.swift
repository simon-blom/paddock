import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native tools management") @MainActor
struct IntegrationsTests {
  static func row() throws -> NativeConnector {
    try JSONDecoder().decode(
      NativeConnector.self, from: JSONSerialization.data(withJSONObject: IntegrationFixture.row))
  }
  @Test func credentialsAreInputOnlyAndExplicitClearIsDifferentFromKeep() throws {
    let keep = ConnectorDraft(row: try Self.row())
    var clear = keep
    clear.headers = [:]
    let kept = String(decoding: try JSONEncoder().encode(keep), as: UTF8.self)
    let cleared = String(decoding: try JSONEncoder().encode(clear), as: UTF8.self)
    #expect(!kept.contains("headers"))
    #expect(cleared.contains("\"headers\":{}"))
    let settings = IntegrationOperation.saveSearch(
      port: 12481, revision: "fixture-revision", provider: "brave", key: nil)
    #expect(
      !String(decoding: try JSONEncoder().encode(settings), as: UTF8.self).contains("\"key\""))
  }
  @Test func checkNeverSavesAndEditingInvalidatesItsResult() async throws {
    let client = IntegrationFixture()
    let editor = ConnectorEditor(client: client, row: try Self.row())
    await editor.check()
    #expect(editor.checked && editor.tools.count == 1)
    #expect(await client.saves == 0)
    editor.draft.url = "https://different.invalid/mcp"
    #expect(!editor.checked)
    await editor.cancel()
  }
  @Test func failurePreservesReviewAndSuccessfulSaveRefreshesOnce() async throws {
    let client = IntegrationFixture()
    let model = IntegrationsModel(client: IntegrationFixture())
    let editor = ConnectorEditor(client: client, row: try Self.row())
    editor.credentialMode = "header"
    editor.headerFields[0].value = "synthetic-secret"
    await client.failWrites(true)
    await editor.save()
    #expect(editor.error != nil && editor.headerFields[0].value == "synthetic-secret")
    await client.failWrites(false)
    var saved = 0
    editor.onSaved = { saved += 1 }
    await editor.save()
    #expect(saved == 1 && editor.headerFields[0].value.isEmpty)
    await model.refresh()
    #expect(model.rows.count == 1)
  }
  @Test func multipleCredentialHeadersRequireUniqueNames() throws {
    let editor = ConnectorEditor(client: IntegrationFixture(), row: try Self.row())
    editor.credentialMode = "header"
    editor.headerFields[0].value = "first"
    editor.headerFields.append(.init(name: "authorization", value: "second"))
    #expect(!editor.valid)
    editor.headerFields[1].name = "X-Api-Key"
    #expect(editor.valid && editor.request.headers?.count == 2)
  }
  @Test func failedCheckNeedsExplicitOverrideAndEditsInvalidateIt() async throws {
    let client = IntegrationFixture()
    await client.failChecks(true)
    let editor = ConnectorEditor(client: client, row: try Self.row())
    await editor.save()
    #expect(editor.saveAnyway && !editor.checked)
    #expect(await client.saves == 0)
    editor.draft.url = "https://corrected.invalid/mcp"
    #expect(!editor.saveAnyway)
    await editor.save()
    #expect(await client.saves == 0)
    await editor.save()
    #expect(await client.saves == 1)
  }
  @Test func scopeLivesInTheSameReviewAndSavePreservesOtherEndpoints() async throws {
    let client = IntegrationFixture()
    let editor = ConnectorEditor(client: client, row: try Self.row())
    editor.scopePorts = [12481, 12482]
    await editor.save()
    #expect(await client.scopePorts == [12481, 12482])
    #expect(await client.checks == 1)
    #expect(await client.saves == 1)
  }
  @Test func endpointDraftsProtectPortAndSaveThroughTheSharedScopeCommand() async throws {
    let client = IntegrationFixture()
    let model = IntegrationsModel(client: client)
    await model.refresh()
    await model.loadSearch(12481)
    model.setEndpoint(try Self.row(), port: 12481, enabled: true)
    await model.loadSearch(12482)
    #expect(model.searchEditor?.settings.port == 12481)
    await model.saveEndpoint(12481)
    #expect(await client.scopePorts == [12481])
    #expect(model.endpointChanges.isEmpty)
  }
  @Test func connectorManagementBelongsOnlyToManager() {
    #expect(!StudioDestination.allCases.map(\.rawValue).contains("Connectors"))
    #expect(ManagerDestination.allCases.contains(.connectors))
    #expect(!ManagerDestination.allCases.map(\.rawValue).contains("Tools & search"))
  }
  @Test func composerManageConnectorsSwitchesAreasWithoutLosingStudioState() {
    let workspace = WorkspaceModel(client: IntegrationFixture())
    workspace.navigation.studio = .chats
    workspace.navigation.sidebarVisible = false
    workspace.draft.message = "An unsent question"

    workspace.integrations.onManage?()

    #expect(workspace.navigation.mode == .manager)
    #expect(workspace.navigation.manager == .connectors)
    #expect(workspace.navigation.sidebarVisible)
    #expect(workspace.draft.message == "An unsent question")
    workspace.navigation.mode = .studio
    #expect(workspace.navigation.studio == .chats)
    #expect(!workspace.navigation.sidebarVisible)
  }
  @Test func failedScopeAndDeleteKeepSavedRows() async throws {
    let client = IntegrationFixture()
    let model = IntegrationsModel(client: IntegrationFixture())
    let tested = IntegrationsModel(client: client)
    await tested.refresh()
    await client.failWrites(true)
    let row = try Self.row()
    #expect(!(await tested.write(.remove(id: row.id, revision: row.revision))))
    #expect(tested.rows.count == 1 && tested.error != nil && !tested.saving)
    await model.stop()
  }
  @Test func searchConfigurationProtectsDraftAndPreservesKeyOnFailure() async throws {
    let client = IntegrationFixture()
    let model = IntegrationsModel(client: IntegrationFixture())
    let settings = try JSONDecoder().decode(
      SearchConfiguration.self,
      from: Data(#"{"port":12481,"revision":"fixture","provider":"brave","hasKey":true}"#.utf8))
    let editor = WebSearchEditor(client: client, settings: settings)
    editor.provider = "exa"
    editor.key = "synthetic-key"
    await client.failWrites(true)
    await editor.save()
    #expect(editor.dirty && editor.key == "synthetic-key" && editor.error != nil)
    model.searchEditor = editor
    await model.loadSearch(12482)
    #expect(model.searchEditor === editor)
    editor.reset()
    #expect(!editor.dirty && editor.provider == "brave")
  }
  @Test func searchKeyValidationKeepsExistingKeysButRequiresOneWhenChangingProvider() async throws {
    let settings = try JSONDecoder().decode(
      SearchConfiguration.self,
      from: Data(#"{"port":12481,"revision":"fixture","provider":"brave","hasKey":true}"#.utf8))
    let editor = WebSearchEditor(client: IntegrationFixture(), settings: settings)
    #expect(editor.keepsSavedKey && editor.validation == nil && !editor.dirty)
    editor.provider = "tavily"
    #expect(!editor.keepsSavedKey && editor.validation?.contains("Tavily") == true)
    await editor.save()
    #expect(editor.error != nil && editor.dirty)
    editor.key = "synthetic-key"
    #expect(editor.validation == nil)
    editor.reset()
    #expect(editor.validation == nil && editor.key.isEmpty && !editor.dirty)
    editor.provider = ""
    #expect(editor.validation == nil && editor.dirty)
  }
  @Test func settingsChoicesWrapWithoutGapsOverflowOrReordering() {
    let sizes = [60, 90, 110, 100, 85, 115].map { CGSize(width: $0, height: 32) }
    for width: CGFloat in [230, 420, 700] {
      let rects = SettingsChoiceLayout.positions(sizes: sizes, width: width, spacing: 8)
      #expect(rects.count == 6)
      for (i, rect) in rects.enumerated() {
        #expect(rect.maxX <= width && rect.size == sizes[i])
        if i > 0 {
          let previous = rects[i - 1]
          #expect(rect.minY == previous.minY ? rect.minX == previous.maxX + 8 : rect.minX == 0)
        }
      }
    }
  }
  @Test func toolsSettingsRenderLongMCPNamesKeysAndEmptyStatesOffscreen() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      for width: CGFloat in [520, 820] {
        let model = IntegrationsModel(client: IntegrationFixture())
        let settings = try JSONDecoder().decode(
          SearchConfiguration.self,
          from: Data(#"{"port":12481,"revision":"fixture","provider":"brave","hasKey":true}"#.utf8))
        let editor = WebSearchEditor(client: IntegrationFixture(), settings: settings)
        model.rows = try [false, true].enumerated().map { index, system in
          var row = IntegrationFixture.row
          row["id"] = "fixture-\(index)"
          row["system"] = system
          row["label"] =
            system
            ? "Shared research tools"
            : "Documentation and knowledge search with a deliberately long server name"
          row["url"] =
            "https://example.invalid/very/long/server/endpoint/that/must/not/push/the/toggle/outside/the/window/mcp"
          return try JSONDecoder().decode(
            NativeConnector.self, from: JSONSerialization.data(withJSONObject: row))
        }
        let root = ScrollView {
          VStack(alignment: .leading, spacing: 20) {
            EndpointFormCard("Web search") { WebSearchForm(editor: editor) }
            EndpointMCPSection(model: model, port: 12481)
          }.padding(24).frame(maxWidth: .infinity)
        }.background(PaddockStyle.canvas).environment(\.colorScheme, dark ? .dark : .light)
        let host = NSHostingController(rootView: root)
        host.sizingOptions = []
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: width, height: 850),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentViewController = host
        window.setContentSize(NSSize(width: width, height: 850))
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        for state in ["saved", "missing-key", "empty"] {
          if state == "missing-key" { editor.provider = "exa" }
          if state == "empty" {
            editor.provider = ""
            model.rows = []
          }
          try await Task.sleep(for: .milliseconds(100))
          host.view.layoutSubtreeIfNeeded()
          #expect(abs(host.view.frame.width - width) < 1)
          let bitmap = try #require(host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds))
          host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
          if let directory = ProcessInfo.processInfo.environment["PADDOCK_ENDPOINT_SNAPSHOTS"] {
            try bitmap.representation(using: .png, properties: [:])?.write(
              to: URL(fileURLWithPath: directory)
                .appending(path: "tools-\(state)-\(Int(width))-\(dark ? "dark" : "light").png"))
          }
        }
      }
    }
  }
  @Test func formsRenderOffscreenInBothThemes() throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      let client = IntegrationFixture()
      let row = try Self.row()
      let editor = ConnectorEditor(client: client, row: row)
      editor.credentialMode = "header"
      editor.headerFields[0].value = "synthetic-secret"
      let views = [
        AnyView(ConnectorReviewView(editor: editor, onCancel: {})),
        AnyView(
          WebSearchForm(
            editor: WebSearchEditor(
              client: client,
              settings: try JSONDecoder().decode(
                SearchConfiguration.self,
                from: Data(
                  #"{"port":12481,"revision":"fixture","provider":"brave","hasKey":true}"#.utf8)))
          ).frame(width: 522, alignment: .leading)),
        AnyView(ConnectorSignInView(model: IntegrationsModel(client: client), row: row)),
      ]
      for view in views {
        let host = NSHostingView(rootView: view.environment(\.colorScheme, dark ? .dark : .light))
        host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        host.frame = NSRect(x: 0, y: 0, width: 570, height: 720)
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width <= 580)
        let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        #expect(bitmap.pixelsWide > 0)
      }
    }
  }
  @Test func catalogRendersOffscreenWithoutAnEmptyFirstScreen() throws {
    _ = NSApplication.shared
    for width in [760.0, 1100.0] {
      for dark in [false, true] {
        let model = IntegrationsModel(client: IntegrationFixture())
        model.hits = try (0..<8).map { n in
          let fixture: [String: Any] = [
            "key": "fixture-\(n)",
            "name": ["GitHub", "Documentation", "Very long example connector name"][n % 3],
            "description": "Synthetic connector", "domain": "example.invalid",
            "authorityTier": n % 2 == 0 ? "S" : "A",
            "liveness": n % 2 == 0 ? "ok" : "auth-required", "toolCount": 12, "githubStars": 250,
            "remoteEndpoints": [],
          ]
          return try JSONDecoder().decode(
            ConnectorHit.self, from: JSONSerialization.data(withJSONObject: fixture))
        }
        let view = IntegrationsView(model: model, endpoints: []).environment(
          \.colorScheme, dark ? .dark : .light)
        let host = NSHostingView(rootView: view)
        host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        host.frame = NSRect(x: 0, y: 0, width: width, height: 640)
        host.layoutSubtreeIfNeeded()
        let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        #expect(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0)
        if let directory = ProcessInfo.processInfo.environment["PADDOCK_TOOLS_SNAPSHOTS"],
          directory.hasPrefix("/tmp/paddock-tools-parity-surfaces"),
          let png = bitmap.representation(using: .png, properties: [:])
        {
          try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
          try png.write(
            to: URL(fileURLWithPath: directory).appending(
              path: "catalog-\(Int(width))-\(dark ? "dark" : "light").png"))
        }
      }
    }
  }
}

private actor IntegrationFixture: ManagerLoading {
  static var row: [String: Any] {
    [
      "id": "00000000-0000-0000-0000-000000000001", "label": "fixture",
      "url": "https://example.invalid/mcp", "registryKey": "", "system": false, "ports": [],
      "revision": 1, "hasHeaders": true, "connected": false, "keychain": true, "oauthRevision": 0,
    ]
  }
  var saves = 0
  var checks = 0
  var scopePorts: [UInt16] = []
  private var checkFailure = false
  func failChecks(_ value: Bool) { checkFailure = value }
  private var fail = false
  func failWrites(_ value: Bool) { fail = value }
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.core("Not used") }
  func integrations(_ command: IntegrationCommand) async throws -> IntegrationReply {
    var value: [String: Any] = [:]
    var failure = false
    if case .run(let op) = command {
      failure = fail && op.mutation
      switch op {
      case .list: value = ["connectors": [Self.row]]
      case .check:
        checks += 1
        failure = checkFailure
        value = ["tools": [["name": "fixture_tool", "description": "Fixture only"]]]
      case .scope(_, _, _, let ports): scopePorts = ports
      case .searchSettings(let port):
        value = ["search": ["port": port, "revision": "fixture", "provider": "", "hasKey": false]]
      case .save:
        saves += 1
        value = ["savedId": Self.row["id"]!, "message": "Saved"]
      default: break
      }
    }
    let reply: [String: Any] = [
      "job": [
        "id": "fixture", "status": failure ? "failed" : "succeeded",
        "message": failure ? "Synthetic failure" : "", "value": value,
      ]
    ]
    return try JSONDecoder().decode(
      IntegrationReply.self, from: JSONSerialization.data(withJSONObject: reply))
  }
}
