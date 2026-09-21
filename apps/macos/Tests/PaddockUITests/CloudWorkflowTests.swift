import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Cloud provider workflows") @MainActor
struct CloudWorkflowTests {
  @Test func lockedAccountsRequireExplicitUnlockAndRefreshWithoutAProviderProbe() async throws {
    let row = try CloudWorkflowFixture.row(credentialReady: false)
    let client = CloudWorkflowFixture(rows: [row])
    let model = ConnectionsModel(client: client)
    var refreshes = 0
    model.onChange = { refreshes += 1 }
    await model.refresh()
    #expect(model.rows.first?.credentialReady == false)
    #expect(await client.unlocks == 0)
    await model.unlock(row)
    #expect(model.rows.first?.credentialReady == true)
    #expect(model.error == nil && !model.busy)
    #expect(await client.unlocks == 1)
    #expect(await client.checks == 0)
    #expect(refreshes == 1)
  }
  @Test func knownProviderFormsUseExactAddressesAndFormats() throws {
    let client = CloudWorkflowFixture()
    for service in CloudService.allCases where service != .custom {
      let editor = ConnectionEditor(
        client: client, connection: nil, openRouter: false, pick: nil, service: service)
      #expect(editor.draft.baseUrl == service.base)
      #expect(editor.draft.kind == service.kind)
      #expect(editor.draft.name == service.rawValue)
      #expect(editor.openRouter == (service == .openrouter))
      #expect(!editor.canCheck)
      editor.key = "synthetic-key"
      #expect(editor.canCheck)
    }
    let lookalike = try CloudWorkflowFixture.row(
      id: "gateway", base: "https://api.openai.com.evil.invalid/v1", kind: "openai")
    #expect(CloudService.service(for: lookalike) == .custom)
    let gateway = try CloudWorkflowFixture.row(
      id: "gateway", base: "https://gateway.invalid/v1", kind: "anthropic")
    #expect(CloudService.service(for: gateway) == .custom)
  }

  @Test func inlineAddsKeepAutoAndProviderPinsDistinctAndPreserveNewerPicks() async throws {
    let row = try CloudWorkflowFixture.row()
    let client = CloudWorkflowFixture(rows: [row])
    let model = ConnectionsModel(client: client)
    await model.refresh()
    var refreshes = 0
    model.onChange = { refreshes += 1 }
    let entry = try #require(try await CloudFixture().catalog().models.first)
    let provider = try #require(try await CloudFixture().providers(for: entry.id).providers.first)
    let auto = CloudModelPick(model: entry, provider: nil)
    let pin = CloudModelPick(model: entry, provider: provider)
    await model.add(auto, to: row, service: .openrouter)
    await model.add(pin, to: row, service: .openrouter)  // old captured row must not erase auto
    await model.add(pin, to: row, service: .openrouter)  // duplicate click is not a write
    #expect(model.editor == nil)
    #expect(model.rows.first?.models.map(\.pickKey) == [auto.pickKey, pin.pickKey])
    #expect(await client.modelWrites == 2)
    #expect(await client.checks == 0)
    #expect(refreshes == 2)
    await model.removeModel(auto, from: row)
    #expect(model.rows.first?.models == [pin])
    #expect(model.rows.first?.models.first?.ctx == provider.ctx)
    #expect(refreshes == 3)
  }

  @Test func missingOrFailedKeysOpenReviewAndFailedWritesRetainPicks() async throws {
    let row = try CloudWorkflowFixture.row()
    let client = CloudWorkflowFixture(rows: [row])
    let model = ConnectionsModel(client: client)
    await model.refresh()
    let pick = CloudModelPick(id: "fixture/model")
    await client.failWrites()
    await model.add(pick, to: row, service: .openrouter)
    #expect(model.error != nil && model.rows.first?.models.isEmpty == true)
    await client.failChecks()
    await model.test(row)
    #expect(model.checks[row.id]?.ok == false)
    await model.add(pick, to: row, service: .openrouter)
    #expect(model.editor?.models == [pick])
    await model.cancelReview()
    await model.add(pick, to: nil, service: .anthropic)
    #expect(model.editor?.service == .anthropic)
    #expect(model.editor?.draft.baseUrl == CloudService.anthropic.base)
  }

  @Test func savedCatalogsAreAccountAndRevisionScoped() async throws {
    let a = try CloudWorkflowFixture.row(id: "a", base: CloudService.openai.base, kind: "openai")
    let b = try CloudWorkflowFixture.row(id: "b", base: CloudService.openai.base, kind: "openai")
    let client = CloudWorkflowFixture(rows: [a, b])
    let model = ConnectionsModel(client: client)
    await model.refresh()
    let first = model.catalog(for: a)
    #expect(model.catalog(for: a) === first)
    let second = model.catalog(for: b)
    #expect(first !== second)
    await first.refresh()
    await second.refresh()
    #expect(first.models.first?.id == "a/model")
    #expect(second.models.first?.id == "b/model")
    #expect(await client.cancels == 2)
    model.selectedAccounts[.openai] = b.id
    #expect(model.account(for: .openai)?.id == b.id)
    await model.add(CloudModelPick(id: "b/model"), to: b, service: .openai)
    let updated = try #require(model.account(for: .openai))
    #expect(model.catalog(for: updated) !== second)
    #expect(model.catalog(for: updated).models.first?.id == "b/model")
    await model.catalog(for: updated).loadIfNeeded()
    #expect(
      await client.checks == 2, "A pick-only save must not reload the catalog or reset search")
    #expect(first === model.catalog(for: a))
    let beforeExternalEdit = model.catalog(for: updated)
    try await client.advanceRevision(b.id)
    await model.refresh()
    let edited = try #require(model.account(for: .openai))
    let invalidated = model.catalog(for: edited)
    #expect(
      invalidated !== beforeExternalEdit && invalidated.models.isEmpty,
      "An external key-only revision must not inherit the old catalog")
    await invalidated.refresh()
    #expect(await client.checks == 3)
    await model.remove(a)
    await model.add(CloudModelPick(id: "gone/model"), to: a, service: .openai)
    #expect(model.editor == nil && model.error != nil)
  }

  @Test func cancelledCatalogReleasesItsCheckAndNeverWrites() async throws {
    let client = CloudWorkflowFixture(delay: true)
    let browser = SavedCloudCatalog(client: client, connection: try CloudWorkflowFixture.row())
    let task = Task { try await browser.catalog() }
    while await client.checks == 0 { await Task.yield() }
    task.cancel()
    await client.finishCheck()
    do {
      _ = try await task.value
      Issue.record("Cancelled browse unexpectedly completed")
    } catch { #expect(error is CancellationError) }
    #expect(await client.cancels == 1)
    #expect(await client.modelWrites == 0)
  }

  @Test func sparseListsDoNotAdvertiseMissingPriceContextOrDates() throws {
    let bare = try JSONDecoder().decode(CloudModel.self, from: Data(#"{"id":"bare-model"}"#.utf8))
    #expect(
      CloudCatalogPresentation.orders(for: [bare], ranked: false) == [.name, .nameDescending])
    #expect(CloudCatalogPresentation.manualPick("  my-model  ")?.id == "my-model")
    for invalid in ["", "\n", "x@y", "cloud:x", "line\nbreak", String(repeating: "x", count: 257)] {
      #expect(CloudCatalogPresentation.manualPick(invalid) == nil)
    }
  }
}

actor CloudWorkflowFixture: ManagerLoading {
  private var rows: [CloudConnection]
  private let delay: Bool
  private var pendingCheck: CheckedContinuation<Void, Never>?
  private var failedCheck = false
  private var failedWrite = false
  private(set) var checks = 0
  private(set) var cancels = 0
  private(set) var modelWrites = 0
  private(set) var unlocks = 0
  init(rows: [CloudConnection] = [], delay: Bool = false) {
    self.rows = rows
    self.delay = delay
  }
  func failChecks() { failedCheck = true }
  func failWrites() { failedWrite = true }
  func advanceRevision(_ id: String) throws {
    let index = try #require(rows.firstIndex { $0.id == id })
    let previous = rows[index]
    rows[index] = try Self.row(
      id: id, base: previous.baseUrl, kind: previous.kind,
      revision: previous.revision + 1, models: previous.models)
  }
  func finishCheck() {
    pendingCheck?.resume()
    pendingCheck = nil
  }
  func snapshot() async throws -> ManagerSnapshot { throw ManagerError.core("No fleet in fixture") }
  static func row(
    id: String = "account", base: String = ConnectionDraft.openRouterBase,
    kind: String = "openai-compat", revision: UInt64 = 1, models: [CloudModelPick] = [],
    credentialReady: Bool = true
  ) throws -> CloudConnection {
    let picks = try JSONSerialization.jsonObject(with: JSONEncoder().encode(models))
    return try JSONDecoder().decode(
      CloudConnection.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": id, "name": "Fixture \(id)", "kind": kind, "baseUrl": base, "hasKey": true,
        "revision": revision, "allowUnauthenticated": false, "credentialStorage": "keychain",
        "models": picks, "credentialReady": credentialReady,
      ]))
  }
  private func object(_ row: CloudConnection) throws -> [String: Any] {
    [
      "id": row.id, "name": row.name, "kind": row.kind, "baseUrl": row.baseUrl,
      "hasKey": row.hasKey,
      "credentialReady": row.credentialReady ?? true,
      "revision": row.revision, "allowUnauthenticated": row.allowUnauthenticated,
      "credentialStorage": row.credentialStorage,
      "models": try JSONSerialization.jsonObject(with: JSONEncoder().encode(row.models)),
    ]
  }
  private func reply(_ object: [String: Any]) throws -> ConnectionReply {
    try JSONDecoder().decode(
      ConnectionReply.self, from: JSONSerialization.data(withJSONObject: object))
  }
  func connections(_ command: ConnectionCommand) async throws -> ConnectionReply {
    switch command {
    case .unlock(let id, let revision):
      let index = try #require(rows.firstIndex { $0.id == id && $0.revision == revision })
      let previous = rows[index]
      rows[index] = try Self.row(
        id: id, base: previous.baseUrl, kind: previous.kind,
        revision: revision, models: previous.models)
      unlocks += 1
      return try reply([
        "job": [
          "id": "unlock-\(unlocks)", "status": "unlocked",
          "message": "Ready", "models": [],
        ]
      ])
    case .list: return try reply(["connections": rows.map { try object($0) }])
    case .check(let draft):
      checks += 1
      #expect(draft.apiKey == nil)  // stored credentials are not fetched by Swift
      if delay { await withCheckedContinuation { pendingCheck = $0 } }
      return try reply([
        "job": [
          "id": "check-\(checks)", "status": failedCheck ? "failed" : "checked",
          "message": failedCheck ? "Synthetic authentication refusal" : "Checked",
          "models": [["id": "\(draft.id ?? "new")/model"]],
        ]
      ])
    case .cancel:
      cancels += 1
      return try reply([:])
    case .models(let id, let revision, let models):
      if failedWrite { throw ManagerError.core("Synthetic write refusal") }
      let index = try #require(rows.firstIndex { $0.id == id })
      #expect(rows[index].revision == revision)
      rows[index] = try Self.row(
        id: id, base: rows[index].baseUrl, kind: rows[index].kind, revision: revision + 1,
        models: models)
      modelWrites += 1
      return try reply([
        "job": ["id": "save-\(modelWrites)", "status": "saved", "message": "Saved", "models": []]
      ])
    case .remove(let id, _):
      rows.removeAll { $0.id == id }
      return try reply([
        "job": ["id": "remove", "status": "saved", "message": "Saved", "models": []]
      ])
    default: throw ManagerError.core("Unexpected fixture command")
    }
  }
}
