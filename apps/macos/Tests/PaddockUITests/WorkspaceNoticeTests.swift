import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Workspace notices", .timeLimit(.minutes(1))) @MainActor
struct WorkspaceNoticeTests {
  @Test(arguments: ["start", "stop", "save", "restart", "remove", "create"])
  func modelRowsOwnTheirLifecycleProgress(action: String) async throws {
    let model = WorkspaceModel(client: NoticeLoader(action: action))
    await model.start()
    model.navigation.showManager(.runners)
    #expect(model.managementNotice == nil)
    let snapshot = try #require(model.snapshot)
    let row = try #require(EndpointRow.rows(snapshot: snapshot, latestJob: model.latestJob).first)
    #expect(row.job?.isActive == true)
    #expect(row.job?.message == NoticeLoader.message)
    #expect(row.status != "Running")
    await model.shutdown()
  }

  @Test func progressWithoutAnAssignedPortRemainsVisible() async throws {
    let model = WorkspaceModel(client: NoticeLoader(port: nil, action: "create"))
    await model.start()
    model.navigation.showManager(.runners)
    #expect(model.managementNotice?.isActive == true)
    await model.shutdown()
  }

  @Test func failuresAlwaysKeepTheirExplanationAndCompletedJobsDisappear() async throws {
    for state in ["failed", "succeeded"] {
      let model = WorkspaceModel(client: NoticeLoader(state: state))
      await model.start()
      model.navigation.showManager(.runners)
      #expect((model.managementNotice != nil) == (state == "failed"))
      if state == "failed" { #expect(model.managementNotice?.message == NoticeLoader.message) }
      await model.shutdown()
    }
  }

  @Test func otherPagesDoNotLoseProgressWithoutAnInlineStatus() async throws {
    let model = WorkspaceModel(client: NoticeLoader())
    await model.start()
    #expect(model.navigation.mode == .studio)
    #expect(model.managementNotice != nil)
    model.navigation.showManager(.models)
    #expect(model.managementNotice != nil)
    model.navigation.showManager(.runners)
    model.detailEndpointPort = 12345
    #expect(model.managementNotice != nil)
    model.detailEndpointPort = nil
    model.systemToolsPort = 12345
    #expect(model.managementNotice != nil)
    model.systemToolsPort = nil
    #expect(model.managementNotice == nil)
    await model.shutdown()
  }
}

private struct NoticeLoader: ManagerLoading {
  static let message = "Checking the saved endpoint and available memory, then loading the model…"
  var port: UInt16? = 12345
  var action = "start"
  var state = "running"

  func snapshot() async throws -> ManagerSnapshot {
    let base = try endpointFixture()
    var object: [String: Any] = [
      "id": 1, "action": action, "state": state, "message": Self.message,
    ]
    if let port { object["port"] = Int(port) }
    let job = try ManagerWire.decode(
      ManagementJob.self, from: JSONSerialization.data(withJSONObject: object))
    return ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: base.catalog,
      runners: base.runners, servers: base.servers ?? [], jobs: [job])
  }
}
