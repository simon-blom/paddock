import Foundation
import Testing

@testable import PaddockConversationCore

struct VisionBudgetTests {
  typealias V = ConversationValue
  static func fixture() throws -> V {
    let url = try #require(
      Bundle.module.url(
        forResource: "vision-budget", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(V.self, from: Data(contentsOf: url))
  }
  @Test func webAndNativeUseIdenticalTowerContracts() throws {
    let fixture = try Self.fixture()
    for row in try #require(fixture["cases"]?.array) {
      let budget = try #require(
        NativeVisionBudget(value: fixture["budgets"]?[row["family"]!.string!]))
      #expect(
        budget.tokens(
          width: row["width"]?.integer, height: row["height"]?.integer,
          detail: row["detail"]!.string!) == row["expected"]?.integer)
    }
  }
  @Test func absentInvalidAndUnmeasuredBudgetsNeverInventACost() throws {
    let raw = try #require(Self.fixture()["budgets"]?["bonsai"]?.object)
    let budget = try #require(NativeVisionBudget(value: .object(raw)))
    #expect(NativeVisionBudget(value: nil) == nil)
    #expect(NativeVisionBudget(value: .object([:])) == nil)
    for key in ["pixels_per_token", "min_tokens", "auto_max_tokens", "max_pixels", "max_edge"] {
      var bad = raw
      bad[key] = .number(0)
      #expect(NativeVisionBudget(value: .object(bad)) == nil)
    }
    for dimension: Int? in [nil, 0, -1, Int.max] {
      #expect(budget.tokens(width: dimension, height: 100, detail: "auto") == nil)
    }
    #expect(budget.tokens(width: 100, height: 100, detail: "invalid") == nil)
  }
}
