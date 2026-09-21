import Foundation
import PaddockClient
import Testing

@testable import PaddockConversationCore

struct ComparePresentationTests {
  typealias O = [String: ConversationValue]
  func fixtures(_ name: String) throws -> [O] {
    let url = try #require(
      Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode([O].self, from: Data(contentsOf: url))
  }
  @Test func cloudIdentityMatchesTheActualWebStoreIncludingProviderPins() throws {
    for row in try fixtures("compare-identity") {
      let result = CloudModelIdentity.resolve(
        id: row["id"]!.string!, display: row["display"]?.string, kind: row["kind"]?.string ?? "",
        provider: row["provider"]?.string)
      #expect(result.name == row["name"]?.string)
      #expect(result.vendor == row["vendor"]?.string, "\(row["id"]!.string!)")
    }
  }
  @Test func removedCloudConnectionStillHasMakerAndReadableFallbackWithoutUUID() {
    let id = "cloud:old-account:moonshotai/kimi-k3@deepinfra/bf16"
    let bare = CloudModelIdentity.bareModel(id)
    #expect(bare == "moonshotai/kimi-k3")
    #expect(CloudModelIdentity.vendor(bare) == "Moonshot")
    #expect(CloudModelIdentity.fallbackName(id) == "kimi k3")
    #expect(CloudModelIdentity.fallbackName("/models/Qwen3.8-27B-Q4_K_M.gguf") == "Qwen3.8 27B")
  }
  @Test func raceBadgesMatchWebAndDoNotRewardIncompleteOrContendedRuns() throws {
    for row in try fixtures("compare-races") {
      let actual = NativeComparePresentation.fastest(
        row["messages"]?.array?.compactMap(\.object) ?? [])
      #expect(actual == row["expected"]?.string, "\(row["name"]!.string!)")
    }
  }
  @Test func newTimestampMatchesIntegerMillisecondsExpectedByTheSharedStore() throws {
    let encoded = try JSONEncoder().encode(NativeStudioRuntime.now)
    let n = try JSONDecoder().decode(Int64.self, from: encoded)
    #expect(abs(Double(n) - Date().timeIntervalSince1970 * 1000) < 1000)
  }
}
