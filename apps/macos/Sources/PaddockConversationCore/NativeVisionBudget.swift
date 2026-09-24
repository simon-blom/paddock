import Foundation

/// The runner's tower contract, mirrored from studio/src/lib/vision.ts.
/// These are estimates, not tokenizer measurements. Unknown/malformed contracts
/// must not manufacture a cost or prevent a request to an older/cloud server.
public struct NativeVisionBudget: Codable, Sendable, Equatable {
  public let maxPixels: Int
  public let minPixels: Int
  public let maxEdge: Int?
  public let pixelsPerToken: Int
  public let maxTokens: Int
  public let minTokens: Int
  public let autoMaxTokens: Int
  private enum CodingKeys: String, CodingKey {
    case maxPixels = "max_pixels"
    case minPixels = "min_pixels"
    case maxEdge = "max_edge"
    case pixelsPerToken = "pixels_per_token"
    case maxTokens = "max_tokens"
    case minTokens = "min_tokens"
    case autoMaxTokens = "auto_max_tokens"
  }

  public init?(value: ConversationValue?) {
    guard let fields = value?.object,
      let maxPixels = fields["max_pixels"]?.integer,
      let minPixels = fields["min_pixels"]?.integer,
      let pixelsPerToken = fields["pixels_per_token"]?.integer,
      let maxTokens = fields["max_tokens"]?.integer,
      let minTokens = fields["min_tokens"]?.integer,
      let autoMaxTokens = fields["auto_max_tokens"]?.integer
    else { return nil }
    let edge = fields["max_edge"]
    guard edge == nil || edge == .null || edge?.integer != nil else { return nil }
    self.maxPixels = maxPixels
    self.minPixels = minPixels
    self.maxEdge = edge?.integer
    self.pixelsPerToken = pixelsPerToken
    self.maxTokens = maxTokens
    self.minTokens = minTokens
    self.autoMaxTokens = autoMaxTokens
    guard valid else { return nil }
  }

  private var valid: Bool {
    minPixels > 0 && maxPixels >= minPixels && maxPixels <= 9_007_199_254_740_991
      && pixelsPerToken > 0 && pixelsPerToken <= maxPixels
      && minTokens > 0 && maxTokens >= minTokens && maxTokens <= Int(Int32.max)
      && autoMaxTokens >= minTokens && autoMaxTokens <= maxTokens
      && (maxEdge.map { $0 > 0 && $0 <= Int(UInt32.max) } ?? true)
  }

  public func tokens(width: Int?, height: Int?, detail: String) -> Int? {
    guard valid, let width, let height, width > 0, height > 0,
      width <= Int(UInt32.max), height <= Int(UInt32.max),
      ["auto", "high", "low"].contains(detail)
    else { return nil }
    let cap = detail == "high" ? maxTokens : detail == "low" ? minTokens : autoMaxTokens
    let maxPixels = min(
      Double(maxPixels), max(Double(minPixels), Double(cap) * Double(pixelsPerToken)))
    var w = Double(width)
    var h = Double(height)
    if let maxEdge, max(w, h) > Double(maxEdge) {
      let scale = Double(maxEdge) / max(w, h)
      w *= scale
      h *= scale
    }
    if w * h > maxPixels {
      let scale = sqrt(maxPixels / (w * h))
      w *= scale
      h *= scale
    }
    let rows = ceil(max(1, floor(w)) * max(1, floor(h)) / Double(pixelsPerToken))
    return Int(min(Double(maxTokens), max(Double(minTokens), rows)))
  }
}

extension NativeStudioRuntime {
  /// Validate before saving/consuming attachments. Changing models or sending
  /// through a keyboard/menu action cannot bypass the composer's size check.
  func validateImageBudget(_ parts: [V], models laneIDs: [String]) throws {
    for id in laneIDs {
      let cap = capability(id)
      guard cap["vision"]?.bool == true, let context = cap["max_ctx"]?.integer, context > 0,
        let budget = NativeVisionBudget(value: cap["vision_budget"])
      else { continue }
      let tokens = parts.reduce(0) { sum, part in
        guard part["type"]?.string == "image" else { return sum }
        return sum
          + (budget.tokens(
            width: part["width"]?.integer, height: part["height"]?.integer,
            detail: part["detail"]?.string ?? "auto") ?? 0)
      }
      if tokens >= context {
        let name = models.first { $0["id"]?.string == id }?["title"]?.string ?? id
        throw ConversationFailure.invalid(
          "Images alone need approximately \(tokens.formatted()) tokens; \(name) has a \(context.formatted())-token context. Choose Auto-resize or Smaller, or remove an image."
        )
      }
    }
  }
}
