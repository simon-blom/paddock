import Foundation
import Testing

@testable import PaddockClient

@Suite("Tools ABI contract")
struct IntegrationWireTests {
  @Test func unlockIsExplicitRevisionBoundAndCarriesNoCredentials() throws {
    let operation = IntegrationOperation.unlock(id: "fixture", revision: 7)
    #expect(operation.mutation)
    let json = try #require(
      try JSONSerialization.jsonObject(with: JSONEncoder().encode(operation)) as? [String: Any])
    #expect(Set(json.keys) == ["kind", "id", "revision"])
    #expect(json["kind"] as? String == "unlock")
    #expect(json["revision"] as? Int == 7)
  }
  @Test(
    .enabled(if: ProcessInfo.processInfo.environment["PADDOCK_INTEGRATION_TEST_LIBRARY"] != nil))
  func isolatedCorePersistsTheSameLibraryAndRefusesStaleMutation() async throws {
    let env = ProcessInfo.processInfo.environment
    let root = try #require(env["PADDOCK_DATA"])
    #expect(root.hasPrefix("/tmp/paddock-integrations-core."))
    guard root.hasPrefix("/tmp/paddock-integrations-core.") else { return }
    let library = URL(fileURLWithPath: try #require(env["PADDOCK_INTEGRATION_TEST_LIBRARY"]))
    let client = NativeManager(libraryURL: library)
    #expect(try await client.integration(.list).connectors?.isEmpty == true)
    var draft = ConnectorDraft()
    draft.label = "synthetic-fixture"
    draft.url = "https://example.invalid/mcp"
    let saved = try await client.integration(.save(draft))
    let id = try #require(saved.savedId)
    let row = try #require(try await client.integration(.list).connectors?.first)
    #expect(
      row.id == id && row.revision == 1 && !row.hasHeaders && !row.system && row.ports.isEmpty)
    await client.close()
    let reopened = NativeManager(libraryURL: library)
    let durable = try #require(try await reopened.integration(.list).connectors?.first)
    #expect(durable == row)
    #expect(durable.credentialReady == true)
    _ = try await reopened.integration(.unlock(id: id, revision: durable.revision))
    var edit = ConnectorDraft(row: durable)
    edit.label = "renamed-fixture"
    _ = try await reopened.integration(.save(edit))
    do {
      _ = try await reopened.integration(.remove(id: id, revision: 1))
      Issue.record("A stale native removal was accepted")
    } catch { #expect(error.localizedDescription.contains("changed")) }
    let updated = try #require(try await reopened.integration(.list).connectors?.first)
    #expect(updated.label == "renamed-fixture" && updated.revision == 2)
    _ = try await reopened.integration(.remove(id: id, revision: 2))
    #expect(try await reopened.integration(.list).connectors?.isEmpty == true)
    await reopened.close()
  }
}
