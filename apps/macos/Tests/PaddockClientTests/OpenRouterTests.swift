import Foundation
import Testing

@testable import PaddockClient

@Suite("Native OpenRouter contracts", .timeLimit(.minutes(2)))
struct OpenRouterTests {
  @Test func normalizedPricesAndProviderVariantsArePreserved() throws {
    let model = try ManagerWire.decode(
      CloudModel.self,
      from: Data(
        #"{"id":"qwen/test","maxOut":8192,"promptPrice":0.0000002,"completionPrice":null,"tools":true}"#
          .utf8))
    #expect(model.maxOut == 8192)
    #expect(model.promptPrice == 0.0000002)
    #expect(model.completionPrice == nil)
    #expect(model.tools == true)
    #expect(model.free == nil)
    let result = try ManagerWire.decode(
      CloudProviders.self,
      from: Data(
        #"{"providers":[{"name":"Same brand","tag":"brand/eu","promptPrice":0},{"name":"Same brand","tag":"brand/us","promptPrice":0.000003}]}"#
          .utf8))
    #expect(result.providers.count == 2)
    #expect(result.providers.map(\.tag) == ["brand/eu", "brand/us"])
    #expect(result.providers.first?.promptPrice == 0)
    let chosen = CloudModelPick(model: model, provider: result.providers[1])
    #expect(chosen.pickKey == "qwen/test@brand/us")
    #expect(chosen.maxOut == 8192)
    let wire = try #require(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(chosen)) as? [String: Any])
    #expect(wire["provider"] as? String == "brand/us")
    #expect(wire["maxOut"] as? Int == 8192)
    #expect(wire["promptPrice"] == nil)
    let automatic = CloudModelPick(model: model, provider: nil)
    #expect(automatic.pickKey == "qwen/test")
    #expect(automatic.provider == nil)
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_DESKTOP_TEST_LIBRARY"] != nil))
  func invalidRequestFailsBeforeNetworkingOrOpeningUserState() async throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_DESKTOP_TEST_LIBRARY"])
    let client = NativeOpenRouter(libraryURL: URL(fileURLWithPath: path))
    await #expect(throws: ManagerError.core("Invalid OpenRouter model identifier")) {
      try await client.providers(for: "qwen/test?key=test-only")
    }
  }

  // Explicit opt-in diagnostic, not part of the hermetic check script. Only
  // anonymous catalog GETs; never a key check, inference call or model download.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OPENROUTER_LIVE"] == "1"))
  func livePublicCatalogThroughBundledABI() async throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_DESKTOP_TEST_LIBRARY"])
    let client = NativeOpenRouter(libraryURL: URL(fileURLWithPath: path))
    let catalog = try await client.catalog()
    #expect(catalog.ranked)
    #expect(catalog.models.count > 10)
    let model = try #require(catalog.models.first { $0.id.hasPrefix("qwen/") && $0.asr != true })
    let result = try await client.providers(for: model.id)
    #expect(!result.providers.isEmpty)
    #expect(result.providers.allSatisfy { !$0.name.isEmpty })
  }
}
