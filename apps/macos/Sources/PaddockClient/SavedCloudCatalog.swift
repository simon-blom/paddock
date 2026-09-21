import Foundation

/// Browse an authenticated account through the existing bounded Rust check.
/// Stored keys never leave Rust. A browse consumes and then cancels its review;
/// it cannot save an endpoint, even when its Swift view is cancelled mid-flight.
public struct SavedCloudCatalog: OpenRouterLoading {
  let client: any ManagerLoading
  let connection: CloudConnection

  public init(client: any ManagerLoading, connection: CloudConnection) {
    self.client = client
    self.connection = connection
  }

  public func catalog() async throws -> CloudCatalog {
    let checked = try await check()
    return CloudCatalog(models: checked.models, ranked: false)
  }

  public func check() async throws -> ConnectionJob {
    try Task.checkCancellation()
    let reply = try await client.connections(.check(ConnectionDraft(connection: connection)))
    guard let receipt = reply.job else {
      throw ManagerError.core("The provider check was not accepted.")
    }
    do {
      var job = receipt
      while job.active {
        try Task.checkCancellation()
        try await Task.sleep(for: .milliseconds(200))
        guard let next = try await client.connections(.poll(job.id)).job else {
          throw ManagerError.core("The provider check expired. Try again.")
        }
        job = next
      }
      try Task.checkCancellation()
      guard job.status == "checked" else { throw ManagerError.core(job.message) }
      await discard(receipt.id)
      return job
    } catch {
      await discard(receipt.id)
      throw error
    }
  }

  private func discard(_ id: String) async {
    // A cancelled parent must still release the credential-bearing receipt.
    let client = client
    await Task { _ = try? await client.connections(.cancel(id)) }.value
  }

  public func providers(for model: String) async throws -> CloudProviders {
    CloudProviders(providers: [])
  }
}
