import Foundation
import Observation
import PaddockClient

/// App-lifetime, bounded ephemeral cache. Navigation does not refetch the
/// catalog. Refresh failures retain the last successful data and timestamp.
@MainActor @Observable
public final class CloudBrowserModel {
  public private(set) var models: [CloudModel] = []
  public private(set) var ranked = false
  public private(set) var refreshedAt: Date?
  public private(set) var loading = false
  public private(set) var error: String?
  private(set) var providerModel: String?
  private(set) var providers: [CloudProvider] = []
  private(set) var providersAt: Date?
  private(set) var providersLoading = false
  private(set) var providersError: String?
  // Window navigation keeps pending per-model provider choices. The connection
  // review captures a stable pick and persists it only after explicit Save.
  private(set) var selectedProviders: [String: String] = [:]
  @ObservationIgnored private let client: any OpenRouterLoading
  @ObservationIgnored private var providerGeneration: UInt64 = 0
  @ObservationIgnored private var stopped = false
  @ObservationIgnored private var catalogTask: Task<CloudCatalog, any Error>?
  @ObservationIgnored private var cache: [String: (rows: [CloudProvider], date: Date)] = [:]

  public init(client: any OpenRouterLoading = NativeOpenRouter()) { self.client = client }

  /// Only for our own confirmed pick-only mutation: the catalog has not changed
  /// credentials or destination. Keep the list steady while rebinding its loader
  /// to the new revision. External edits/key replacements must not use this path.
  func inheritCatalog(from previous: CloudBrowserModel) {
    guard models.isEmpty, !stopped else { return }
    models = previous.models
    ranked = previous.ranked
    refreshedAt = previous.refreshedAt
  }

  func selectedProvider(for model: String) -> CloudProvider? {
    guard let id = selectedProviders[model] else { return nil }
    let rows = providerModel == model ? providers : cache[model]?.rows ?? []
    return rows.first { $0.routingID == id }
  }

  func canSelect(_ provider: CloudProvider, for model: String) -> Bool {
    providerModel == model && !providersLoading
      && providers.filter { $0.routingID == provider.routingID }.count == 1
  }

  @discardableResult
  func selectProvider(_ id: String?, for model: String) -> Bool {
    guard !stopped, models.contains(where: { $0.id == model }) else { return false }
    if let id {
      guard let provider = providers.first(where: { $0.routingID == id }),
        canSelect(provider, for: model)
      else { return false }
      selectedProviders[model] = id
    } else {
      selectedProviders.removeValue(forKey: model)
    }
    return true
  }

  func studioPick(for model: CloudModel) -> CloudModelPick? {
    let chosen = selectedProvider(for: model.id)
    // A vanished provider must never silently become an auto-routed choice.
    if selectedProviders[model.id] != nil && chosen == nil { return nil }
    return CloudModelPick(model: model, provider: chosen)
  }

  func loadIfNeeded() async {
    if let refreshedAt, Date().timeIntervalSince(refreshedAt) < 300 { return }
    await refresh()
  }

  func refresh() async {
    guard !loading, !stopped else { return }
    loading = true
    error = nil
    defer {
      loading = false
      catalogTask = nil
    }
    do {
      // The catalog belongs to the app, not a particular view task. Leaving
      // and re-entering Connections while it loads must not discard its result
      // and strand the replacement view with an empty, non-loading catalog.
      let client = client
      let task = Task { try await client.catalog() }
      catalogTask = task
      let result = try await task.value
      guard !stopped else { return }
      var seen: Set<String> = []
      models = result.models.filter { !$0.id.isEmpty && seen.insert($0.id).inserted }
      ranked = result.ranked
      refreshedAt = Date()
    } catch {
      if !stopped && !(error is CancellationError) { self.error = error.localizedDescription }
    }
  }

  func loadProviders(_ id: String?, force: Bool = false) async {
    guard !stopped else { return }
    providerGeneration &+= 1
    let attempt = providerGeneration
    providerModel = id
    providers = []
    providersAt = nil
    providersError = nil
    providersLoading = false
    guard let id else { return }
    if let cached = cache[id] {
      providers = cached.rows
      providersAt = cached.date
      if !force && Date().timeIntervalSince(cached.date) < 300 { return }
    }
    providersLoading = true
    defer { if attempt == providerGeneration { providersLoading = false } }
    do {
      // Arrow-key navigation need not fire a network request for every row.
      if !force { try await Task.sleep(for: .milliseconds(180)) }
      try Task.checkCancellation()
      guard attempt == providerGeneration else { return }
      let result = try await client.providers(for: id)
      try Task.checkCancellation()
      guard attempt == providerGeneration, !stopped else { return }
      providers = result.providers
      let received = Date()
      providersAt = received
      if cache.count >= 64, let oldest = cache.min(by: { $0.value.date < $1.value.date })?.key {
        cache.removeValue(forKey: oldest)
      }
      cache[id] = (providers, received)
    } catch {
      guard attempt == providerGeneration, !stopped else { return }
      if !(error is CancellationError) { providersError = error.localizedDescription }
    }
  }

  func stop() {
    stopped = true
    catalogTask?.cancel()
    providerGeneration &+= 1
    providersLoading = false
    // Stateless Rust requests have a fixed deadline. No user database or
    // management pointer is held by these in-flight background operations.
  }
}
