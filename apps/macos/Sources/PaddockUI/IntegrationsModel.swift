import Foundation
import Observation
import PaddockClient

@MainActor @Observable
public final class IntegrationsModel {
  var rows: [NativeConnector] = []
  var hits: [ConnectorHit] = []
  var detail: ConnectorHit?
  var details: [String: ConnectorHit] = [:]
  var detailLoading = Set<String>()
  var detailErrors: [String: String] = [:]
  var addingKey: String?
  var query = ""
  var loading = false
  var searching = false
  var busy = false
  var error: String?
  var catalogError: String?
  var message: String?
  var editor: ConnectorEditor?
  var searchEditor: WebSearchEditor?
  var endpointChanges: [String: Bool] = [:]
  var endpointSaving = false
  var selectedID: String?
  var signInRow: NativeConnector?
  var authorization: ConnectorAuthorization?
  var oauthClientID = ""
  @ObservationIgnored let client: any ManagerLoading
  @ObservationIgnored var onChange: (() async -> Void)?
  @ObservationIgnored var onManage: (() -> Void)?
  @ObservationIgnored var onConfigureSearch: ((UInt16) -> Void)?
  @ObservationIgnored var prepareSignIn: (() async -> Void)?
  @ObservationIgnored private var stopped = false
  @ObservationIgnored private var searchGeneration: UInt64 = 0
  @ObservationIgnored private var refreshGeneration: UInt64 = 0
  @ObservationIgnored private var writeTask: Task<Bool, Never>?
  var saving: Bool {
    busy || endpointSaving || editor?.saving == true || searchEditor?.saving == true
  }
  var hasDraft: Bool {
    editor != nil || signInRow != nil || searchEditor?.dirty == true || !endpointChanges.isEmpty
  }
  public init(client: any ManagerLoading) { self.client = client }

  func refresh() async {
    guard !stopped else { return }
    refreshGeneration &+= 1
    let ticket = refreshGeneration
    loading = true
    defer { if ticket == refreshGeneration { loading = false } }
    do {
      let value = try await client.integration(.list)
      guard !stopped, ticket == refreshGeneration else { return }
      guard let rows = value.connectors else {
        throw ManagerError.core("Connector library is unavailable.")
      }
      self.rows = rows
      error = nil
    } catch { if !stopped, ticket == refreshGeneration { self.error = error.localizedDescription } }
  }
  func search() async {
    searchGeneration &+= 1
    let ticket = searchGeneration
    searching = true
    defer { if ticket == searchGeneration { searching = false } }
    do {
      try await Task.sleep(for: .milliseconds(300))
      let value = try await client.integration(.search(query))
      try Task.checkCancellation()
      guard !stopped, ticket == searchGeneration else { return }
      hits = value.results ?? []
      catalogError = nil
    } catch is CancellationError {} catch {
      if !stopped, ticket == searchGeneration { catalogError = error.localizedDescription }
    }
  }
  func loadDetail(_ hit: ConnectorHit) async {
    guard details[hit.key] == nil, !detailLoading.contains(hit.key) else { return }
    detailLoading.insert(hit.key)
    detailErrors[hit.key] = nil
    defer { detailLoading.remove(hit.key) }
    detail = hit
    do {
      let value = try await client.integration(.detail(hit.key))
      try Task.checkCancellation()
      if !stopped {
        if details.count >= 100 { details.removeAll() }
        details[hit.key] = value.detail ?? hit
        if detail?.key == hit.key { detail = value.detail ?? hit }
      }
    } catch is CancellationError {} catch {
      detailErrors[hit.key] = error.localizedDescription
    }
  }
  func review(_ row: NativeConnector? = nil, hit: ConnectorHit? = nil, endpoint: String? = nil) {
    guard !saving, editor == nil, !stopped else { return }
    let editor = ConnectorEditor(client: client, row: row)
    if let hit {
      editor.draft.url = endpoint ?? ""
      editor.draft.registryKey = hit.key
      editor.draft.label = ConnectorCatalog.slug(hit)
      editor.authHint = hit.connection?.authRequired == true || hit.liveness == "auth-required"
    }
    editor.onSaved = { [weak self, weak editor] in
      guard let self, self.editor === editor else { return }
      self.selectedID = editor?.savedID
      self.message = editor?.message
      self.editor = nil
      await self.refresh()
      await self.onChange?()
    }
    self.editor = editor
  }
  func add(_ hit: ConnectorHit) async {
    guard addingKey == nil, !saving, !ConnectorCatalog.added(hit, rows: rows),
      hit.liveness != "dead"
    else { return }
    addingKey = hit.key
    defer { addingKey = nil }
    await loadDetail(hit)
    let resolved = details[hit.key] ?? hit
    guard let url = ConnectorCatalog.endpoint(resolved) else {
      review(hit: resolved)
      editor?.error = "No HTTP endpoint was supplied. Enter the publisher's MCP server URL."
      return
    }
    if resolved.connection?.authRequired == true || resolved.liveness == "auth-required" {
      review(hit: resolved, endpoint: url)
      return
    }
    var draft = ConnectorDraft()
    draft.label = ConnectorCatalog.slug(resolved)
    draft.url = url
    draft.registryKey = resolved.key
    if !(await write(.save(draft))) {
      let failure = error
      review(hit: resolved, endpoint: url)
      editor?.error = failure
    }
  }
  func cancelReview() async {
    guard let editor, !saving else { return }
    if signInRow != nil { await cancelSignIn() }
    guard signInRow == nil else { return }
    await editor.cancel()
    self.editor = nil
  }
  @discardableResult func write(_ operation: IntegrationOperation) async -> Bool {
    guard !saving, !stopped, operation.mutation else { return false }
    busy = true
    error = nil
    // App-owned unstructured task settles despite view/task cancellation.
    let task = Task { [self] in
      defer {
        busy = false
        writeTask = nil
      }
      do {
        let value = try await client.integration(operation)
        authorization = value.authorization ?? authorization
        message = value.message
        await refresh()
        if case .disconnect(let id, _) = operation, let row = rows.first(where: { $0.id == id }) {
          editor?.acceptAuthorizationRevision(row)
        }
        if case .cancelSignIn(let id) = operation, let row = rows.first(where: { $0.id == id }) {
          editor?.acceptAuthorizationRevision(row)
        }
        await onChange?()
        return true
      } catch {
        self.error = error.localizedDescription
        return false
      }
    }
    writeTask = task
    return await task.value
  }
  func beginSignIn() async {
    guard let row = signInRow, !saving else { return }
    await prepareSignIn?()
    _ = await write(
      .signIn(
        id: row.id, revision: row.revision, clientID: oauthClientID.isEmpty ? nil : oauthClientID))
  }
  func cancelSignIn() async {
    guard !saving, let row = signInRow else { return }
    if authorization != nil, !(await write(.cancelSignIn(row.id))) { return }
    authorization = nil
    signInRow = nil
    oauthClientID = ""
  }
  func pollSignIn() async {
    guard let authorization else { return }
    let until = ContinuousClock.now + .seconds(600)
    while !Task.isCancelled, signInRow != nil, ContinuousClock.now < until, !stopped {
      do { try await Task.sleep(for: .seconds(2)) } catch { return }
      await refresh()
      if let row = rows.first(where: { $0.id == authorization.connectorId }),
        row.revision != authorization.revision
      {
        if row.connected, row.oauthRevision > (signInRow?.oauthRevision ?? row.oauthRevision) {
          editor?.acceptAuthorizationRevision(row)
          message = "Signed in."
          self.authorization = nil
          signInRow = nil
          oauthClientID = ""
          await onChange?()
        } else {
          error = "This connector changed during sign-in. Cancel and try again."
        }
        return
      }
    }
    if !Task.isCancelled, !stopped { error = "Sign-in expired. Cancel and start again." }
  }
  func loadSearch(_ port: UInt16) async {
    guard searchEditor?.dirty != true, endpointChanges.isEmpty, !saving else { return }
    if searchEditor?.settings.port == port { return }
    searchEditor = nil
    do {
      guard let settings = try await client.integration(.searchSettings(port)).search else {
        throw ManagerError.core("Search settings unavailable.")
      }
      try Task.checkCancellation()
      guard !stopped else { return }
      let editor = WebSearchEditor(client: client, settings: settings)
      editor.onSaved = { [weak self] in await self?.onChange?() }
      searchEditor = editor
      error = nil
    } catch is CancellationError {} catch { self.error = error.localizedDescription }
  }
  func endpointEnabled(_ row: NativeConnector, port: UInt16) -> Bool {
    row.system || (endpointChanges[row.id] ?? row.ports.contains(port))
  }
  func setEndpoint(_ row: NativeConnector, port: UInt16, enabled: Bool) {
    guard !saving, !row.system else { return }
    endpointChanges[row.id] = enabled == row.ports.contains(port) ? nil : enabled
  }
  func discardEndpoint() {
    guard !saving else { return }
    endpointChanges = [:]
    searchEditor?.reset()
  }
  func saveEndpoint(_ port: UInt16) async {
    guard !saving, let searchEditor, searchEditor.settings.port == port,
      searchEditor.validation == nil
    else { return }
    endpointSaving = true
    error = nil
    let changes = endpointChanges
    let reviewed = rows
    let work = Task { [self] in
      defer {
        endpointSaving = false
        writeTask = nil
      }
      if searchEditor.dirty {
        await searchEditor.save()
        if searchEditor.error != nil { return false }
      }
      do {
        for (id, enabled) in changes.sorted(by: { $0.key < $1.key }) {
          guard let row = reviewed.first(where: { $0.id == id }), !row.system else {
            throw ManagerError.core("A connector changed. Reload before changing its model scope.")
          }
          var ports = Set(row.ports)
          if enabled { ports.insert(port) } else { ports.remove(port) }
          _ = try await client.integration(
            .scope(id: id, revision: row.revision, all: false, ports: ports.sorted()))
          endpointChanges[id] = nil
        }
        message = "Changes saved."
      } catch {
        self.error =
          "Some settings may have been saved. \(error.localizedDescription) Unsaved connector choices are kept; review before retrying."
      }
      // Scope writes change the same TOML revision as search. Refresh it after
      // publication so the next edit does not start from our own stale digest.
      if let current = try? await client.integration(.searchSettings(port)).search {
        searchEditor.settings = current
      }
      let failure = self.error
      await refresh()
      self.error = failure
      await onChange?()
      return failure == nil
    }
    writeTask = work
    _ = await work.value
  }
  func stop() async {
    _ = await writeTask?.value
    if let row = signInRow, authorization != nil {
      _ = await Task { try? await client.integration(.cancelSignIn(row.id)) }.value
    }
    stopped = true
    searchGeneration &+= 1
    refreshGeneration &+= 1
    await editor?.cancel()
    await searchEditor?.settle()
    editor = nil
    searchEditor = nil
  }
}

@MainActor @Observable
final class ConnectorEditor: Identifiable {
  struct HeaderField: Identifiable {
    let id = UUID()
    var name = "Authorization"
    var value = ""
  }
  let id = UUID()
  var draft: ConnectorDraft
  let original: NativeConnector?
  var authHint = false
  var scopeAll = false
  var scopePorts = Set<UInt16>()
  private var baselineAll = false
  private var baselinePorts = Set<UInt16>()
  private var failedCheck: ConnectorDraft?
  var credentialMode: String
  var headerFields = [HeaderField()]
  var checking = false
  var saving = false
  var error: String?
  var message: String?
  var tools: [ConnectorTool] = []
  var savedID: String?
  private var checkedDraft: ConnectorDraft?
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var checkTask: Task<Void, Never>?
  @ObservationIgnored private var saveTask: Task<Void, Never>?
  @ObservationIgnored var onSaved: (() async -> Void)?
  init(client: any ManagerLoading, row: NativeConnector?) {
    self.client = client
    original = row
    draft = ConnectorDraft(row: row)
    credentialMode = row == nil ? "none" : "keep"
    scopeAll = row?.system ?? false
    scopePorts = Set(row?.ports ?? [])
    baselineAll = scopeAll
    baselinePorts = scopePorts
  }
  var request: ConnectorDraft {
    var result = draft
    result.label = result.label.trimmingCharacters(in: .whitespacesAndNewlines)
    result.url = result.url.trimmingCharacters(in: .whitespacesAndNewlines)
    switch credentialMode {
    case "keep": result.headers = nil
    case "header":
      result.headers = Dictionary(
        headerFields.map { ($0.name, $0.value) }, uniquingKeysWith: { first, _ in first })
    default: result.headers = [:]
    }
    return result
  }
  var checked: Bool { checkedDraft == request }
  var saveAnyway: Bool { failedCheck == request }
  var authorizationEditable: Bool {
    guard let original else { return false }
    return draft.url == original.url && draft.label == original.label && credentialMode == "keep"
      && !saving && !checking
  }
  func acceptAuthorizationRevision(_ row: NativeConnector) {
    guard row.id == draft.id, row.url == draft.url else { return }
    draft.revision = row.revision
    checkedDraft = nil
    failedCheck = nil
  }
  var valid: Bool {
    !request.label.isEmpty && !request.url.isEmpty
      && (credentialMode != "header"
        || (!headerFields.isEmpty && headerFields.count <= 16
          && headerFields.allSatisfy { !$0.name.isEmpty && !$0.value.isEmpty }
          && Set(headerFields.map { $0.name.lowercased() }).count == headerFields.count))
  }
  func check() async {
    guard !checking, !saving, valid else { return }
    checking = true
    error = nil
    checkedDraft = nil
    tools = []
    message = nil
    let captured = request
    let task = Task { [self] in
      defer {
        checking = false
        checkTask = nil
      }
      do {
        let value = try await client.integration(.check(captured))
        try Task.checkCancellation()
        guard request == captured else { return }
        checkedDraft = captured
        failedCheck = nil
        authHint = value.authRequired == true
        tools = value.tools ?? []
        message = value.message
      } catch is CancellationError {} catch {
        guard request == captured else { return }
        failedCheck = captured
        self.error = error.localizedDescription
      }
    }
    checkTask = task
    await task.value
  }
  func save() async {
    guard !saving, !checking, valid else { return }
    // Same check-then-save / explicit Save anyway contract as the web form.
    if !checked && !saveAnyway {
      await check()
      guard checked else { return }
    }
    saving = true
    error = nil
    let captured = request
    let task = Task { [self] in
      defer {
        saving = false
        saveTask = nil
      }
      do {
        let value = try await client.integration(.save(captured))
        savedID = value.savedId
        message = value.message
        guard let id = savedID,
          let row = try await client.integration(.list).connectors?.first(where: { $0.id == id })
        else {
          throw ManagerError.core(
            "Saved, but could not reload the connector. Reopen the library before editing again.")
        }
        draft = ConnectorDraft(row: row)
        credentialMode = "keep"
        headerFields = [HeaderField()]
        if scopeAll != baselineAll || scopePorts != baselinePorts {
          _ = try await client.integration(
            .scope(
              id: id, revision: row.revision,
              all: scopeAll, ports: scopeAll ? [] : scopePorts.sorted()))
          baselineAll = scopeAll
          baselinePorts = scopePorts
        }
        checkedDraft = nil
        await onSaved?()
      } catch {
        self.error =
          savedID == nil
          ? error.localizedDescription
          : "The connector was saved, but its remaining settings could not be completed. \(error.localizedDescription)"
      }
    }
    saveTask = task
    await task.value
  }
  func cancel() async {
    if let saveTask { await saveTask.value }
    checkTask?.cancel()
    await checkTask?.value
    headerFields = [HeaderField()]
    checkedDraft = nil
  }
}

@MainActor @Observable
final class WebSearchEditor {
  var settings: SearchConfiguration
  var provider: String
  var key = ""
  var replaceKey = false
  var saving = false
  var error: String?
  var message: String?
  @ObservationIgnored private let client: any ManagerLoading
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored var onSaved: (() async -> Void)?
  init(client: any ManagerLoading, settings: SearchConfiguration) {
    self.client = client
    self.settings = settings
    provider = settings.provider
  }
  var dirty: Bool { provider != settings.provider || replaceKey || !key.isEmpty }
  var providerLabel: String { WebSearchForm.providers.first { $0.0 == provider }?.1 ?? provider }
  var keepsSavedKey: Bool { settings.hasKey && settings.provider == provider && !replaceKey }
  var validation: String? {
    guard !provider.isEmpty else { return nil }
    guard WebSearchForm.providers.contains(where: { $0.0 == provider }) else {
      return "Choose a supported search provider."
    }
    return key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !keepsSavedKey
      ? "Add a \(providerLabel) API key to enable search." : nil
  }
  func reset() {
    guard !saving else { return }
    provider = settings.provider
    key = ""
    replaceKey = false
  }
  func save() async {
    guard !saving, dirty else { return }
    if let validation {
      error = validation
      return
    }
    saving = true
    error = nil
    let operation = IntegrationOperation.saveSearch(
      port: settings.port, revision: settings.revision, provider: provider,
      key: replaceKey || provider != settings.provider || !key.isEmpty ? key : nil)
    let work = Task { [self] in
      defer {
        saving = false
        task = nil
      }
      do {
        let value = try await client.integration(operation)
        message = value.message
        key = ""
        replaceKey = false
        if let current = try await client.integration(.searchSettings(settings.port)).search {
          settings = current
          provider = current.provider
        }
        await onSaved?()
      } catch { self.error = error.localizedDescription }
    }
    task = work
    await work.value
  }
  func settle() async {
    await task?.value
    key = ""
  }
}
