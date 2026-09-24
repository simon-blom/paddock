import Foundation
import PaddockConversationCore

public struct StudioImageEstimate: Sendable {
  public let modelID: String
  public let modelName: String
  public let tokens: Int
  public let context: Int
  public var exceedsContext: Bool { context > 0 && tokens >= context }

  public static func label(_ estimates: [Self]) -> String? {
    guard let low = estimates.map(\.tokens).min(), let high = estimates.map(\.tokens).max() else {
      return nil
    }
    return low == high
      ? "≈ \(low.formatted()) tokens" : "≈ \(low.formatted())–\(high.formatted()) tokens"
  }
}

extension StudioState.Capabilities {
  public struct ImageLane: Decodable, Sendable {
    public let id: String
    public let name: String
    public let context: Int
    public let budget: NativeVisionBudget?
    public let documentParser: Bool?
    public let taskTags: [String]?

    public func tokensForTurn(_ estimates: [Int], instruction: String) -> Int {
      let tag = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
      let curated = [
        "<chart2csv>", "<chart2code>", "<chart2summary>", "<tables_json>", "<tables_html>",
        "<tables_otsl>",
      ]
      let tagged =
        !(taskTags ?? []).isEmpty && ((taskTags ?? []).contains(tag) || curated.contains(tag))
      // Parser/task pages are separate requests, not one combined context.
      return documentParser == true || tagged ? estimates.max() ?? 0 : estimates.reduce(0, +)
    }
  }

  public func imageEstimates(for image: StudioAttachment, detail: String? = nil)
    -> [StudioImageEstimate]
  {
    guard image.mime.hasPrefix("image/") else { return [] }
    return (imageLanes ?? []).compactMap { lane in
      guard
        let tokens = lane.budget?.tokens(
          width: image.width, height: image.height, detail: detail ?? image.detail)
      else { return nil }
      return StudioImageEstimate(
        modelID: lane.id, modelName: lane.name, tokens: tokens, context: lane.context)
    }
  }
}

extension StudioWorkspace {
  public func imageEstimates(for image: StudioAttachment, detail: String? = nil)
    -> [StudioImageEstimate]
  {
    state?.composer?.imageMode == true
      ? [] : state?.capabilities.imageEstimates(for: image, detail: detail) ?? []
  }

  public func attachmentBudgetIssue(for instruction: String) -> String? {
    // Image edits send originals; the image endpoint determines its own
    // reference geometry. Chat's detail/token estimate does not apply.
    guard state?.composer?.imageMode != true else { return nil }
    let estimates = attachments.filter(\.ready).flatMap { imageEstimates(for: $0) }
    for lane in state?.capabilities.imageLanes ?? [] where lane.context > 0 {
      let tokens = lane.tokensForTurn(
        estimates.filter { $0.modelID == lane.id }.map(\.tokens), instruction: instruction)
      if tokens >= lane.context {
        return
          "Images alone need ≈ \(tokens.formatted()) tokens; \(lane.name) has a \(lane.context.formatted())-token context. Choose Auto-resize or Smaller, or remove an image."
      }
    }
    return nil
  }
}
