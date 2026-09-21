import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native connections") @MainActor
struct ConnectionsTests {
  static func row(revision: UInt64 = 1) throws -> CloudConnection {
    try JSONDecoder().decode(
      CloudConnection.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": "00000000-0000-0000-0000-000000000001", "name": "Fixture account",
        "kind": "openai-compat", "baseUrl": ConnectionDraft.openRouterBase,
        "hasKey": true, "revision": revision, "allowUnauthenticated": false,
        "credentialStorage": "keychain",
        "models": [["id": "fixture/model", "provider": "provider/turbo"]],
      ]))
  }
  static func catalogModel() throws -> CloudModel {
    try JSONDecoder().decode(
      CloudModel.self,
      from: Data(#"{"id":"fixture/model","display":"Fixture model","ctx":8192,"vision":true}"#.utf8)
    )
  }
  @Test func wireKeepsKeysInputOnlyAndProviderIdentitiesDistinct() throws {
    var draft = ConnectionDraft(openRouter: true)
    draft.apiKey = "synthetic-secret"
    let encoded = try JSONEncoder().encode(ConnectionCommand.check(draft))
    #expect(String(decoding: encoded, as: UTF8.self).contains("synthetic-secret"))
    let row = try Self.row()
    let pick = try CloudModelPick(model: Self.catalogModel(), provider: nil)
    #expect(row.models[0].pickKey != pick.pickKey)
    let models = try JSONEncoder().encode(
      ConnectionCommand.models(id: row.id, revision: row.revision, models: [pick]))
    #expect(!String(decoding: models, as: UTF8.self).contains("apiKey"))
    #expect(!String(decoding: models, as: UTF8.self).contains("baseUrl"))
    let unchanged = try JSONEncoder().encode(
      ConnectionCommand.check(ConnectionDraft(connection: row)))
    #expect(!String(decoding: unchanged, as: UTF8.self).contains("apiKey"))
  }
  @Test func failedCheckAndSaveRetainDraftAndOriginalProvider() async throws {
    let client = ConnectionFixture(row: try Self.row())
    let editor = ConnectionEditor(
      client: client, connection: try Self.row(), openRouter: true, pick: nil)
    editor.key = "synthetic-new-key"
    await client.setFailure(check: true, save: false)
    await editor.check()
    #expect(!editor.checked && editor.error != nil)
    #expect(editor.key == "synthetic-new-key")
    #expect(editor.models.first?.provider == "provider/turbo")
    await client.setFailure(check: false, save: true)
    await editor.check()
    #expect(editor.checked)
    await editor.save()
    #expect(editor.error != nil && !editor.saving)
    #expect(editor.models.first?.provider == "provider/turbo")
    #expect(editor.key == "synthetic-new-key")
    await editor.cancel()
    #expect(editor.key.isEmpty && editor.job == nil)
  }
  @Test func checkedReviewCannotBeCheckedTwiceAndSaveRefreshesOneSharedModelStore() async throws {
    let client = ConnectionFixture(row: try Self.row())
    let model = ConnectionsModel(client: client)
    await model.refresh()
    var refreshes = 0
    model.onChange = { refreshes += 1 }
    model.review(connection: try Self.row())
    let editor = try #require(model.editor)
    await editor.check()
    await editor.check()
    #expect(await client.checks == 1)
    await editor.save()
    #expect(model.editor == nil && refreshes == 1)
    #expect(await client.saves == 1)
    #expect(model.rows.first?.revision == 2)
  }
  @Test func failedRefreshOrRemovalKeepsSavedRows() async throws {
    let client = ConnectionFixture(row: try Self.row())
    let model = ConnectionsModel(client: client)
    await model.refresh()
    await client.failTransport()
    await model.refresh()
    #expect(model.rows.count == 1 && model.error != nil)
    await model.remove(try Self.row())
    #expect(model.rows.count == 1 && !model.busy && model.error != nil)
  }
  @Test func keylessIsExplicitAndPendingPickIsCapturedBeforeBrowserChanges() async throws {
    let client = ConnectionFixture(row: try Self.row())
    let pick = try CloudModelPick(model: Self.catalogModel(), provider: nil)
    let editor = ConnectionEditor(client: client, connection: nil, openRouter: false, pick: pick)
    editor.draft.name = "Local fixture"
    editor.draft.baseUrl = "http://localhost:9991/v1"
    #expect(!editor.canCheck)
    editor.noAuthentication = true
    #expect(editor.canCheck)
    await editor.check()
    let sent = try #require(await client.lastDraft)
    #expect(sent.allowUnauthenticated && sent.apiKey == "")
    #expect(editor.models == [pick])
    await editor.cancel()
  }
  @Test func nativeFormsRenderOffscreenInBothThemes() throws {
    // Never order a window front or activate the app. This gate produces no
    // desktop fixtures, popovers or focus changes on the user's screen.
    _ = NSApplication.shared
    for dark in [false, true] {
      for openRouter in [false, true] {
        let client = ConnectionFixture(row: try Self.row())
        let editor = ConnectionEditor(
          client: client, connection: openRouter ? try Self.row() : nil, openRouter: openRouter,
          pick: nil)
        let view = NSHostingView(
          rootView: ConnectionReviewView(editor: editor, onCancel: {})
            .environment(\.colorScheme, dark ? .dark : .light))
        view.frame = NSRect(x: 0, y: 0, width: 572, height: 650)
        view.layoutSubtreeIfNeeded()
        #expect(view.fittingSize.width <= 580)
        #expect(view.fittingSize.height < 760)
        let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
        view.cacheDisplay(in: view.bounds, to: bitmap)
        #expect(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0)
      }
    }
  }
}

private actor ConnectionFixture: ManagerLoading {
  private var row: CloudConnection
  private var checkFailure = false
  private var saveFailure = false
  private var transportFailure = false
  private(set) var checks = 0
  private(set) var saves = 0
  private(set) var lastDraft: ConnectionDraft?
  init(row: CloudConnection) { self.row = row }
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.core("No fleet in fixture") }
  func setFailure(check: Bool, save: Bool) {
    checkFailure = check
    saveFailure = save
  }
  func failTransport() { transportFailure = true }
  func connections(_ command: ConnectionCommand) async throws -> ConnectionReply {
    if transportFailure { throw ManagerError.core("Synthetic transport unavailable") }
    switch command {
    case .list:
      let value: [String: Any] = [
        "connections": [
          [
            "id": row.id, "name": row.name, "kind": row.kind, "baseUrl": row.baseUrl,
            "hasKey": row.hasKey, "revision": row.revision, "allowUnauthenticated": false,
            "credentialStorage": "keychain",
            "models": [["id": "fixture/model", "provider": "provider/turbo"]],
          ]
        ]
      ]
      return try decode(value)
    case .check(let draft):
      checks += 1
      lastDraft = draft
      return try job(checkFailure ? "failed" : "checked")
    case .save:
      saves += 1
      if !saveFailure { row = try await ConnectionsTests.row(revision: 2) }
      return try job(saveFailure ? "failed" : "saved")
    case .cancel: return try decode([:])
    default: return try job("saved")
    }
  }
  private func job(_ status: String) throws -> ConnectionReply {
    try decode([
      "job": [
        "id": "fixture-check", "status": status,
        "message": status == "failed" ? "Synthetic refusal" : "Checked", "models": [],
      ]
    ])
  }
  private func decode(_ value: [String: Any]) throws -> ConnectionReply {
    try JSONDecoder().decode(
      ConnectionReply.self, from: JSONSerialization.data(withJSONObject: value))
  }
}
