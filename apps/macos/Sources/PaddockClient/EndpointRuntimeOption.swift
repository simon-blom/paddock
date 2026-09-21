import Foundation

/// Descriptor and saved value come from Rust's reviewed runtime-option schema.
/// No arbitrary config document or saved credential is exposed to Swift.
public struct EndpointRuntimeOption: Decodable, Identifiable, Sendable, Equatable {
  public let id: String
  public let label: String
  public let group: String
  public let kind: String
  public let minimum: Double
  public let maximum: Double
  public let placeholder: String
  public let help: String
  public let capability: String
  public let value: EndpointRuntimeValue?

  public func parse(_ text: String) -> EndpointRuntimeValue? {
    switch kind {
    case "boolean": return text == "true" ? .boolean(true) : text == "false" ? .boolean(false) : nil
    case "integer":
      guard let n = Int64(text), Double(n) >= minimum, Double(n) <= maximum else { return nil }
      return .integer(n)
    case "number":
      guard let n = Double(text), n.isFinite, n >= minimum, n <= maximum else { return nil }
      return .number(n)
    case "text":
      guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        text.utf8.count <= Int(maximum),
        !text.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains)
      else { return nil }
      return .text(text)
    default: return nil
    }
  }
}

public enum EndpointRuntimeValue: Codable, Sendable, Equatable {
  case integer(Int64)
  case number(Double)
  case boolean(Bool)
  case text(String)
  public init(from decoder: any Decoder) throws {
    let c = try decoder.singleValueContainer()
    if let v = try? c.decode(Bool.self) {
      self = .boolean(v)
    } else if let v = try? c.decode(Int64.self) {
      self = .integer(v)
    } else if let v = try? c.decode(Double.self) {
      self = .number(v)
    } else {
      self = .text(try c.decode(String.self))
    }
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.singleValueContainer()
    switch self {
    case .integer(let v): try c.encode(v)
    case .number(let v): try c.encode(v)
    case .boolean(let v): try c.encode(v)
    case .text(let v): try c.encode(v)
    }
  }
  public var text: String {
    switch self {
    case .integer(let v): String(v)
    case .number(let v): String(v)
    case .boolean(let v): v ? "true" : "false"
    case .text(let v): v
    }
  }
}
