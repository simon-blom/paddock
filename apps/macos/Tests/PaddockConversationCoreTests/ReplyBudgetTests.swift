import Foundation
import Testing

@testable import PaddockConversationCore

struct ReplyBudgetTests {
  typealias V = ConversationValue
  @Test func planningHeadroomCannotConsumeASmallModelsEntireContext() {
    #expect(NativeContextPlan.reserve(.null, context: 4096) == 1024)
    #expect(NativeContextPlan.reserve(.number(32768), context: 4096) == 1024)
    #expect(NativeContextPlan.reserve(.number(512), context: 4096) == 512)
    #expect(NativeContextPlan.reserve(.null, context: 131072, outputCeiling: 2048) == 2048)
    #expect(NativeContextPlan.reserve(.number(32768), context: 131072) == 4096)
  }
  @Test func effectiveLimitsMatchTheWebForEveryCapacityAndPreference() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "reply-limit", withExtension: "json", subdirectory: "Fixtures"))
    for row in try JSONDecoder().decode([[String: V]].self, from: Data(contentsOf: url)) {
      let result = NativeReplyBudget.resolve(
        requested: row["requested"]?.integer, cloud: row["cloud"]!.bool!,
        context: row["context"]!.integer!, prompt: row["prompt"]?.integer ?? 0,
        outputCeiling: row["ceiling"]?.integer, exact: row["exact"]?.integer ?? 0)
      #expect(result == row["expected"]?.integer, "\(row["name"]?.string ?? "")")
    }
  }
  @Test func modelMaximumMatchesTheWebPolicyFixtures() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "reply-budget", withExtension: "json", subdirectory: "Fixtures"))
    let cases = try JSONDecoder().decode([[String: V]].self, from: Data(contentsOf: url))
    for row in cases {
      #expect(
        NativeReplyBudget.localMaximum(context: row["context"]!.integer!)
          == (row["context"]!.integer! > 0 ? row["context"]!.integer! : 4096))
      let budget = NativeReplyBudget.maximum(
        context: row["context"]!.integer!, prompt: row["prompt"]!.integer!,
        outputCeiling: row["ceiling"]?.integer, exact: row["exact"]?.integer ?? 0)
      #expect(budget == row["expected"]?.integer, "\(row["name"]?.string ?? "")")
    }
  }
  @Test func estimateIncludesSentThinkingInstructionsAndImageCostButNotBase64() {
    let input: [V] = [
      .object([
        "type": .string("message"),
        "content": .array([
          .object(["type": .string("input_text"), "text": .string("Hero")]),
          .object([
            "type": .string("input_image"),
            "image_url": .string(String(repeating: "A", count: 10000)),
          ]),
          .object([
            "type": .string("input_file"),
            "file_data": .string(String(repeating: "B", count: 10000)),
          ]),
        ]),
      ]),
      .object([
        "type": .string("reasoning"),
        "content": .array([.object(["type": .string("reasoning_text"), "text": .string("Think")])]),
      ]),
    ]
    #expect(
      NativeReplyBudget.estimate(input: input, instructions: "System") == 6 + 4 + 1 + 512 + 4 + 2)
  }
}
