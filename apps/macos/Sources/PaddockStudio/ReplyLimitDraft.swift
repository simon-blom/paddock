import Foundation

/// The same editor contract as Studio's reply-limit.ts. A custom value is
/// exact; model changes never rewrite it. Automatic is represented by null,
/// not a magic token count. Invalid/empty custom drafts must not become null.
public struct ReplyLimitDraft: Equatable, Sendable {
  public static let maximum = 1_048_576
  public var automatic: Bool
  public var text: String

  public init(value: Int? = nil) {
    automatic = value == nil
    text = value.map(String.init) ?? ""
  }
  public var tokens: Int? {
    let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !raw.isEmpty, raw.utf8.allSatisfy({ (48...57).contains($0) }),
      let n = Int(raw), (1...Self.maximum).contains(n)
    else { return nil }
    return n
  }
  public var validation: String? {
    automatic || tokens != nil ? nil : "Enter a whole number from 1 to 1,048,576 tokens."
  }
  public var value: Int? { automatic ? nil : tokens }
}
