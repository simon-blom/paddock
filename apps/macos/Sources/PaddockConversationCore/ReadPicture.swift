import Foundation

/// SQLite history uses the web Studio's content-addressed image table. Runs
/// store only name/ref pairs, so re-reading never duplicates image payloads.
public struct ReadPicture: Equatable, Codable, Sendable, Identifiable {
  public let name: String
  public let url: String
  public var id: String { ref }
  public let ref: String
  public init(name: String, url: String) {
    self.name = name
    self.url = url
    self.ref = Self.reference(url)
  }
  public var historyReference: ConversationValue {
    .object(["name": .string(name), "ref": .string(ref)])
  }
  public static func reference(_ url: String) -> String {
    // cyrb53: identical to studio/src/lib/reads.ts, including UTF-16 length.
    let units = url.utf16
    var h1: UInt32 = 0xdead_beef
    var h2: UInt32 = 0x41c6_ce57
    for c in units {
      h1 = (h1 ^ UInt32(c)) &* 2_654_435_761
      h2 = (h2 ^ UInt32(c)) &* 1_597_334_677
    }
    h1 = ((h1 ^ (h1 >> 16)) &* 2_246_822_507) ^ ((h2 ^ (h2 >> 13)) &* 3_266_489_909)
    h2 = ((h2 ^ (h2 >> 16)) &* 2_246_822_507) ^ ((h1 ^ (h1 >> 13)) &* 3_266_489_909)
    let hash = (UInt64(h2 & 2_097_151) << 32) + UInt64(h1)
    return String(units.count, radix: 36) + "-" + String(hash, radix: 36)
  }
  public static func restore(_ run: ConversationValue, table: ConversationValue?) throws -> [Self] {
    try (run["images"]?.array ?? []).map { item in
      guard let ref = item["ref"]?.string, let url = table?[ref]?.string,
        url.hasPrefix("data:image/"), url.contains(";base64,"),
        reference(url) == ref
      else { throw ConversationFailure.invalid("A saved read image is missing or damaged.") }
      return Self(name: item["name"]?.string ?? "Image", url: url)
    }
  }
}
