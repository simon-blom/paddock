import Foundation
import Testing

@testable import PaddockConversationCore

struct OCRQualityTests {
  @Test func sharesWebGzipReviewFixtures() throws {
    let url = try #require(
      Bundle.module.url(forResource: "ocr-quality", withExtension: "json", subdirectory: "Fixtures")
    )
    let rows = try JSONDecoder().decode(
      [[String: ConversationValue]].self, from: Data(contentsOf: url))
    for row in rows {
      let text = String(repeating: row["text"]!.string!, count: row["repeat"]!.integer!)
      let ratio = try NativeOCRQuality.repetitionRatio(text)
      #expect(
        (ratio > NativeOCRQuality.threshold) == row["review"]!.bool!,
        "\(row["name"]!) ratio \(ratio)")
    }
  }
  @Test func resourceBoundAndCancellationAreExplicit() async throws {
    #expect(throws: ConversationFailure.tooLarge) {
      try NativeOCRQuality.repetitionRatio(String(repeating: "x", count: 4 * 1024 * 1024 + 1))
    }
    let task = Task {
      try await Task.sleep(for: .seconds(30))
      return try NativeOCRQuality.repetitionRatio(String(repeating: "x", count: 1000))
    }
    task.cancel()
    await #expect(throws: CancellationError.self) { try await task.value }
  }
}
