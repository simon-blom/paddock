import Foundation
import Testing

@testable import PaddockClient

@Suite("Connection ABI contract")
struct ConnectionWireTests {
  @Test func unlockOnlyCarriesAccountIdentityAndReviewedRevision() throws {
    let data = try JSONEncoder().encode(
      ConnectionCommand.unlock(id: "fixture-account", revision: 7))
    let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(Set(object.keys) == ["kind", "id", "revision"])
    #expect(object["kind"] as? String == "unlock")
    #expect(object["id"] as? String == "fixture-account")
    #expect(object["revision"] as? Int == 7)
  }
  @Test func missingKeyAndExplicitKeyRemovalHaveDifferentWireMeanings() throws {
    let keep = ConnectionDraft(openRouter: true)
    var clear = keep
    clear.apiKey = ""
    clear.allowUnauthenticated = true
    func object(_ value: ConnectionDraft) throws -> [String: Any] {
      try #require(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
    }
    #expect(try object(keep)["apiKey"] == nil)
    #expect(try object(clear)["apiKey"] as? String == "")
    #expect(try object(clear)["allowUnauthenticated"] as? Bool == true)
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_CONNECTION_TEST_LIBRARY"] != nil))
  func actualNativeABIUsesAnIsolatedCoreAndNeverNeedsWebKit() async throws {
    let env = ProcessInfo.processInfo.environment
    let root = try #require(env["PADDOCK_DATA"])
    #expect(root.hasPrefix("/tmp/paddock-connections-core."))
    guard root.hasPrefix("/tmp/paddock-connections-core.") else { return }
    let path = try #require(env["PADDOCK_CONNECTION_TEST_LIBRARY"])
    let client = NativeManager(libraryURL: URL(fileURLWithPath: path))
    let initial = try await client.connections(.list)
    #expect(initial.connections?.isEmpty == true)
    var draft = ConnectionDraft()
    draft.name = "Invalid URL fixture"
    draft.baseUrl = "file:///tmp/never-read"
    draft.allowUnauthenticated = true
    let receipt = try #require(try await client.connections(.check(draft)).job)
    var result = receipt
    for _ in 0..<100 where result.active {
      try await Task.sleep(for: .milliseconds(10))
      result = try #require(try await client.connections(.poll(receipt.id)).job)
    }
    #expect(result.status == "failed")
    #expect(try await client.connections(.list).connections?.isEmpty == true)
    await client.close()
    // A fresh owned core proves that the reviewed-but-invalid draft did not
    // leak a connection into durable state during any phase.
    let reopened = NativeManager(libraryURL: URL(fileURLWithPath: path))
    #expect(try await reopened.connections(.list).connections?.isEmpty == true)
    await reopened.close()
  }
}
