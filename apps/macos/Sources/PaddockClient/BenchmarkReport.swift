import Foundation

public struct BenchmarkHistory: Decodable, Sendable {
  public let reports: [BenchmarkReport]
}
public struct BenchmarkReport: Decodable, Sendable, Identifiable {
  public let id, model: String
  public let createdAtMs: Int64
  public let concurrency, promptWords, trials, warmups, outputLimit: Int
  public let aggregateOutputTokS, wallSeconds: Double
  public let ttftMedianMs, streamEventGapP99Ms: Double?
  public let outputTokens: Int
  public let maxCtx, maxBatch: Int?
  public let cachePolicy: String
  public let runnerVersion: String?
  public let samples: [Sample]
  public struct Sample: Decodable, Sendable {
    public let inputTokens, outputTokens: Int
    public let cachedTokens: Int?
    public let ttftMs, durationMs: Double
    public let finishReason: String
  }
}
