import Foundation
import Observation
import PaddockClient

/// App-owned connection state. Neither a view disappearing nor an unrelated
/// fleet refresh discards a review or causes a duplicate mutation.
@MainActor @Observable
public final class ConnectionsModel {
  public private(set) var rows: [CloudConnection] = []
  public private(set) var loaded = false
  public private(set) var loading = false
  public private(set) var busy = false
  public private(set) var error: String?
  var editor: ConnectionEditor?
  var selectedOpenRouterID: String?
  var selectedAccounts: [CloudService: String] = [:]
  private(set) var checkingAccount: String?
  private(set) var checks: [String: (revision: UInt64, ok: Bool, message: String)] = [:]
  @ObservationIgnored private var catalogs:
    [String: (revision: UInt64, browser: CloudBrowserModel)] = [:]
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var generation: UInt64 = 0
  @ObservationIgnored private var stopped = false
  @ObservationIgnored var onChange: (() async -> Void)?
  public init(client: any ManagerLoading) { self.client = client }
  var saving: Bool { busy || editor?.saving == true }
  var hasDraft: Bool { editor != nil }

  func refresh() async {
    guard !stopped else { return }
    generation &+= 1
    let ticket = generation
    loading = true
    defer { if ticket == generation { loading = false } }
    do {
      let reply = try await client.connections(.list)
      guard ticket == generation, !stopped else { return }
      guard let rows = reply.connections else {
        throw ManagerError.core("Saved connections are unavailable.")
      }
      self.rows = rows
      for (id, cached) in catalogs
      where !rows.contains(where: { $0.id == id && $0.revision == cached.revision }) {
        cached.browser.stop()
        catalogs.removeValue(forKey: id)
      }
      checks = checks.filter { id, check in
        rows.contains { $0.id == id && $0.revision == check.revision }
      }
      loaded = true
      error = nil
    } catch {
      if ticket == generation, !stopped { self.error = error.localizedDescription }
    }
  }

  func review(
    connection: CloudConnection? = nil, openRouter: Bool = false, pick: CloudModelPick? = nil,
    service: CloudService? = nil
  ) {
    guard !busy, !stopped, editor == nil else { return }
    let value = ConnectionEditor(
      client: client, connection: connection, openRouter: openRouter, pick: pick, service: service)
    if connection == nil, value.service != .custom {
      let names = Set(rows.map(\.name))
      var ordinal = 2
      while names.contains(value.draft.name) {
        value.draft.name = "\(value.service.rawValue) \(ordinal)"
        ordinal += 1
      }
    }
    value.onSaved = { [weak self, weak value] in
      guard let self, self.editor === value else { return }
      if let endpoint = value?.job?.endpoint, endpoint.isOpenRouter {
        self.selectedOpenRouterID = endpoint.id
      }
      if let endpoint = value?.job?.endpoint {
        self.selectedAccounts[CloudService.service(for: endpoint)] = endpoint.id
      }
      self.editor = nil
      await self.refresh()
      await self.onChange?()
    }
    editor = value
  }
  func cancelReview() async {
    guard let editor, !editor.saving else { return }
    await editor.cancel()
    self.editor = nil
  }

  func remove(_ row: CloudConnection) async {
    await mutate(.remove(id: row.id, revision: row.revision))
  }
  func account(for service: CloudService) -> CloudConnection? {
    let accounts = rows.filter { CloudService.service(for: $0) == service }
    let selected =
      selectedAccounts[service] ?? (service == .openrouter ? selectedOpenRouterID : nil)
    return accounts.first { $0.id == selected } ?? accounts.first
  }

  func catalog(for row: CloudConnection) -> CloudBrowserModel {
    if let cached = catalogs[row.id], cached.revision == row.revision { return cached.browser }
    catalogs[row.id]?.browser.stop()
    let browser = CloudBrowserModel(client: SavedCloudCatalog(client: client, connection: row))
    catalogs[row.id] = (row.revision, browser)
    return browser
  }

  func test(_ row: CloudConnection) async {
    guard checkingAccount == nil, !stopped, !busy else { return }
    checkingAccount = row.id
    defer { checkingAccount = nil }
    do {
      let result = try await SavedCloudCatalog(client: client, connection: row).check()
      guard !stopped, rows.contains(where: { $0.id == row.id && $0.revision == row.revision })
      else { return }
      checks[row.id] = (row.revision, true, result.message)
    } catch {
      guard !stopped, !(error is CancellationError),
        rows.contains(where: { $0.id == row.id && $0.revision == row.revision })
      else { return }
      checks[row.id] = (row.revision, false, error.localizedDescription)
    }
  }

  func unlock(_ row: CloudConnection) async {
    guard !busy, !stopped, checkingAccount == nil else { return }
    busy = true
    error = nil
    defer { busy = false }
    do {
      let reply = try await client.connections(.unlock(id: row.id, revision: row.revision))
      guard let job = reply.job else {
        throw ManagerError.core("Cloud authorization was not accepted.")
      }
      let terminal = try await ConnectionEditor.wait(job, client: client)
      _ = try? await client.connections(.cancel(job.id))
      guard !stopped, !Task.isCancelled else { return }
      guard terminal.status == "unlocked" else { throw ManagerError.core(terminal.message) }
      await refresh()
      await onChange?()
    } catch {
      if !stopped, !(error is CancellationError) { self.error = error.localizedDescription }
    }
  }

  func add(_ pick: CloudModelPick, to row: CloudConnection?, service: CloudService) async {
    guard loaded, !saving, editor == nil, !stopped else { return }
    // Resolve the current revision after a preceding save. Never overwrite a
    // newer pick list captured by a now-stale row closure.
    let current = row.flatMap { requested in rows.first { $0.id == requested.id } }
    if row != nil && current == nil {
      error = "This account was removed. Choose an account before adding the model."
      return
    }
    if let current, CloudService.service(for: current) != service {
      error = "The account changed. Select the provider again before adding the model."
      return
    }
    guard let current, current.hasKey || current.allowUnauthenticated,
      checks[current.id]?.ok != false
    else {
      review(connection: current, pick: pick, service: service)
      return
    }
    guard !current.models.contains(where: { $0.pickKey == pick.pickKey }) else { return }
    await mutate(
      .models(id: current.id, revision: current.revision, models: current.models + [pick]))
  }
  func removeModel(_ pick: CloudModelPick, from row: CloudConnection) async {
    guard let current = rows.first(where: { $0.id == row.id }) else { return }
    await mutate(
      .models(
        id: current.id, revision: current.revision,
        models: current.models.filter { $0.pickKey != pick.pickKey }
      ))
  }
  private func mutate(_ command: ConnectionCommand) async {
    guard !busy, !stopped else { return }
    busy = true
    error = nil
    defer { busy = false }
    do {
      var unchangedCatalog: (row: CloudConnection, browser: CloudBrowserModel)?
      if case .models(let id, let revision, _) = command,
        let row = rows.first(where: { $0.id == id && $0.revision == revision }),
        let cached = catalogs[id], cached.revision == revision
      {
        unchangedCatalog = (row, cached.browser)
      }
      let reply = try await client.connections(command)
      guard let job = reply.job else {
        throw ManagerError.core("The connection operation was not accepted.")
      }
      let terminal = try await ConnectionEditor.wait(job, client: client)
      guard terminal.status == "saved" else { throw ManagerError.core(terminal.message) }
      await refresh()
      if let unchangedCatalog,
        let row = rows.first(where: {
          $0.id == unchangedCatalog.row.id && $0.revision == unchangedCatalog.row.revision + 1
            && $0.baseUrl == unchangedCatalog.row.baseUrl && $0.kind == unchangedCatalog.row.kind
        })
      {
        // Any additional revision (including a concurrent key edit) invalidates
        // this exception. A pick-only Save must not blank/reset the browser.
        catalog(for: row).inheritCatalog(from: unchangedCatalog.browser)
      }
      await onChange?()
    } catch { self.error = error.localizedDescription }
  }
  func stop() async {
    stopped = true
    generation &+= 1
    for cached in catalogs.values { cached.browser.stop() }
    await editor?.cancel()
  }
}

@MainActor @Observable
final class ConnectionEditor: Identifiable {
  let id = UUID()
  let original: CloudConnection?
  let openRouter: Bool
  let service: CloudService
  var draft: ConnectionDraft
  var key = ""
  var noAuthentication: Bool
  var models: [CloudModelPick]
  private(set) var discovered: [CloudModel] = []
  private(set) var job: ConnectionJob?
  private(set) var checking = false
  private(set) var saving = false
  private(set) var error: String?
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var attempt: UInt64 = 0
  @ObservationIgnored private var closed = false
  @ObservationIgnored var onSaved: (() async -> Void)?

  init(
    client: any ManagerLoading, connection: CloudConnection?, openRouter: Bool,
    pick: CloudModelPick?, service: CloudService? = nil
  ) {
    self.client = client
    original = connection
    self.service =
      connection.map { CloudService.service(for: $0) }
      ?? service ?? (openRouter ? .openrouter : .custom)
    self.openRouter = self.service == .openrouter
    var initialDraft = ConnectionDraft(connection: connection, openRouter: openRouter)
    if connection == nil, self.service != .custom {
      initialDraft.name = self.service.rawValue
      initialDraft.kind = self.service.kind
      initialDraft.baseUrl = self.service.base
    }
    draft = initialDraft
    noAuthentication = connection?.allowUnauthenticated ?? false
    models = connection?.models ?? []
    if let pick, !models.contains(where: { $0.pickKey == pick.pickKey }) { models.append(pick) }
  }
  var checked: Bool { job?.status == "checked" }
  var canCheck: Bool {
    !checking && !saving && !checked
      && !draft.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && !draft.baseUrl.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      && (noAuthentication || !key.isEmpty || original?.hasKey == true)
  }
  func toggle(_ model: CloudModel) {
    guard !saving else { return }
    let pick = CloudModelPick(model: model, provider: nil)
    if models.contains(where: { $0.pickKey == pick.pickKey }) {
      models.removeAll { $0.pickKey == pick.pickKey }
    } else if models.count < 256 {
      models.append(pick)
    }
  }
  func check() async {
    guard canCheck, !closed else { return }
    checking = true
    error = nil
    attempt &+= 1
    let ticket = attempt
    defer { if ticket == attempt { checking = false } }
    var request = draft
    request.allowUnauthenticated = noAuthentication
    request.apiKey = noAuthentication ? "" : (key.isEmpty ? nil : key)
    do {
      if let job { _ = try? await client.connections(.cancel(job.id)) }
      let reply = try await client.connections(.check(request))
      guard let receipt = reply.job else {
        throw ManagerError.core("The connection check was not accepted.")
      }
      guard ticket == attempt, !closed else {
        _ = try? await client.connections(.cancel(receipt.id))
        return
      }
      job = receipt
      let result = try await Self.wait(receipt, client: client)
      guard ticket == attempt, !closed else { return }
      job = result
      guard result.status == "checked" else { throw ManagerError.core(result.message) }
      discovered = result.models
    } catch {
      if ticket == attempt, !closed { self.error = error.localizedDescription }
    }
  }
  func editDetails() async {
    guard !saving else { return }
    attempt &+= 1
    checking = false
    if let job { _ = try? await client.connections(.cancel(job.id)) }
    job = nil
    error = nil
  }
  func save() async {
    guard checked, let receipt = job, !saving, !closed else { return }
    saving = true
    error = nil
    defer { saving = false }
    do {
      let reply = try await client.connections(.save(receipt: receipt.id, models: models))
      guard let accepted = reply.job else { throw ManagerError.core("The save was not accepted.") }
      let result = try await Self.wait(accepted, client: client)
      job = result
      guard result.status == "saved" else { throw ManagerError.core(result.message) }
      key = ""
      draft.apiKey = nil
      closed = true
      await onSaved?()
    } catch { self.error = error.localizedDescription }
  }
  func cancel() async {
    guard !saving else { return }
    closed = true
    attempt &+= 1
    checking = false
    if let job { _ = try? await client.connections(.cancel(job.id)) }
    job = nil
    key = ""
    draft.apiKey = nil
  }
  /// Once a mutation is accepted, always collect its receipt. Cancelling a
  /// Swift task cannot turn a completed SQLite write into a claimed failure.
  static func wait(_ job: ConnectionJob, client: any ManagerLoading) async throws -> ConnectionJob {
    try await Task {
      var value = job
      while value.active {
        try await Task.sleep(for: .milliseconds(200))
        guard let next = try await client.connections(.poll(value.id)).job else {
          throw ManagerError.core("The operation status is unavailable. Refresh before retrying.")
        }
        value = next
      }
      return value
    }.value
  }
}
