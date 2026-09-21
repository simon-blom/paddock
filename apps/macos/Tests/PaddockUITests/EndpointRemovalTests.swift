import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Remove saved model configurations", .serialized) @MainActor
struct EndpointRemovalTests {
  @Test func onlyConfirmedStoppedConfigurationsAreRemovable() throws {
    let base = try endpointFixture()
    let stopped = try removalEndpoint(12345)
    #expect(
      EndpointRemovalReview(.init(port: 12345, runner: nil, configured: stopped, job: nil)) != nil)
    #expect(
      EndpointRemovalReview(
        .init(port: 12345, runner: base.runners.first, configured: stopped, job: nil)) == nil)
    #expect(
      EndpointRemovalReview(
        .init(
          port: 12345, runner: nil, configured: try removalEndpoint(12345, running: true), job: nil)
      ) == nil)
    #expect(
      EndpointRemovalReview(
        .init(
          port: 12345, runner: nil, configured: try removalEndpoint(12345, revision: nil), job: nil)
      ) == nil)
    #expect(
      EndpointRemovalReview(.init(port: 12345, runner: nil, configured: nil, job: nil)) == nil)
    #expect(
      EndpointRemovalReview(
        .init(port: 12345, runner: nil, configured: stopped, job: try commandJob())) == nil)
  }

  @Test func duplicateNamesRemoveOnlyTheReviewedPortAndReturnToTheList() async throws {
    let client = RemovalFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let row = try #require(
      workspace.snapshot.flatMap { EndpointRow.rows(snapshot: $0, latestJob: nil).last })
    workspace.openEndpoint(row)
    let review = try #require(EndpointRemovalReview(row))
    #expect(review.port == 12346 && review.title == "Qwen 3.8 27B")
    #expect(workspace.canRemoveEndpoint(review))
    #expect(await client.removedPorts.isEmpty, "Reviewing must never delete")
    #expect(await workspace.removeEndpoint(review))
    #expect(await client.removedPorts == [12346])
    #expect(workspace.snapshot?.servers?.map(\.port) == [12345])
    #expect(workspace.detailEndpointPort == nil && workspace.endpointEditor == nil)
    #expect(workspace.snapshot?.catalog.models.first?.installed == true)
    #expect(!workspace.canRemoveEndpoint(review))
    #expect(await workspace.removeEndpoint(review) == false)
    #expect(await client.removedPorts == [12346], "No repeat submission after removal")
    await workspace.shutdown()
  }

  @Test func dirtyDraftsAreProtectedWithoutBlockingAnUnrelatedDuplicate() async throws {
    let client = RemovalFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let rows = EndpointRow.rows(snapshot: try #require(workspace.snapshot), latestJob: nil)
    workspace.openEndpoint(rows[0])
    workspace.endpointEditor?.context = "8192"
    #expect(workspace.endpointEditor?.dirty == true)
    let own = try #require(EndpointRemovalReview(rows[0]))
    #expect(!workspace.canRemoveEndpoint(own))
    #expect(await workspace.removeEndpoint(own) == false)
    #expect(await client.removedPorts.isEmpty)
    let other = try #require(EndpointRemovalReview(rows[1]))
    #expect(workspace.canRemoveEndpoint(other))
    #expect(await workspace.removeEndpoint(other))
    #expect(workspace.endpointEditor?.context == "8192")
    #expect(workspace.detailEndpointPort == 12345)
    await workspace.shutdown()
  }

  @Test func changedRevisionOrStartingRunnerInvalidatesTheConfirmation() async throws {
    for running in [false, true] {
      let client = RemovalFixture()
      let workspace = WorkspaceModel(client: client)
      await workspace.start()
      let row = try #require(
        workspace.snapshot.flatMap { EndpointRow.rows(snapshot: $0, latestJob: nil).first })
      let review = try #require(EndpointRemovalReview(row))
      await client.change(running: running)
      await workspace.refresh()
      #expect(!workspace.canRemoveEndpoint(review))
      #expect(await workspace.removeEndpoint(review) == false)
      #expect(workspace.desktopError != nil && workspace.snapshot?.servers?.count == 2)
      #expect(await client.removedPorts.isEmpty)
      await workspace.shutdown()
    }
  }

  @Test func refusalIsVisibleAndDoesNotRemoveTheRow() async throws {
    let client = RemovalFixture()
    let workspace = WorkspaceModel(client: client)
    await workspace.start()
    let row = try #require(
      workspace.snapshot.flatMap { EndpointRow.rows(snapshot: $0, latestJob: nil).first })
    let review = try #require(EndpointRemovalReview(row))
    await client.refuse()
    #expect(await workspace.removeEndpoint(review) == false)
    #expect(workspace.commandError == "Configuration changed")
    #expect(workspace.snapshot?.servers?.count == 2)
    #expect(await client.removedPorts.isEmpty)
    await workspace.shutdown()
  }
}

private actor RemovalFixture: ManagerLoading {
  private(set) var removedPorts: [UInt16] = []
  private var revision = "reviewed"
  private var running = false
  private var failure = false
  private var jobs: [ManagementJob] = []
  func change(running: Bool) {
    self.running = running
    if !running { revision = "changed" }
  }
  func refuse() { failure = true }
  func snapshot() async throws -> ManagerSnapshot {
    let base = try endpointFixture()
    return ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: base.catalog,
      runners: [],
      servers: try [UInt16(12345), 12346].filter { !removedPorts.contains($0) }
        .map { try removalEndpoint($0, revision: revision, running: running) }, jobs: jobs)
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    guard case .remove(let port, let revision) = command, !failure, !running,
      revision == self.revision
    else { throw ManagerError.core("Configuration changed") }
    let receipt = try ManagerWire.decode(
      ManagementJob.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": 1, "port": Int(port), "action": "remove", "state": "succeeded",
        "message": "Configuration removed",
      ]))
    removedPorts.append(port)
    jobs = [receipt]
    return receipt
  }
}

private func removalEndpoint(_ port: UInt16, revision: String? = "reviewed", running: Bool = false)
  throws -> ConfiguredEndpoint
{
  try ManagerWire.decode(
    ConfiguredEndpoint.self,
    from: JSONSerialization.data(withJSONObject: [
      "port": Int(port), "display": "Qwen 3.8 27B", "model": "qwen3.8-27b", "artifact": "mlx-4bit",
      "revision": revision as Any? ?? NSNull(), "running": running, "local_only": true,
      "settings": [
        "host": "127.0.0.1", "max_ctx": 4096, "max_batch": 1, "device": "metal",
        "has_api_key": true, "vision": false, "forensics": false,
      ],
    ]))
}
