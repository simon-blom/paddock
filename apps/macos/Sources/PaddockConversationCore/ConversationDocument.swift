import Foundation

public struct ConversationDocument: Sendable, Equatable {
  public typealias Value = ConversationValue
  public typealias Object = [String: Value]
  public private(set) var fields: Object
  public var id: String { fields["id"]!.string! }
  public var leafID: String? { fields["leafId"]?.string }
  public var messages: [Object] { fields["messages"]!.array!.map { $0.object! } }
  public var title: String { fields["title"]?.string ?? "" }
  public var branchMemory: [String: String] {
    (fields["branchMemory"]?.object ?? [:]).compactMapValues(\.string)
  }
  public static let maximumBytes = 64 * 1024 * 1024

  public init(data: Data) throws {
    guard data.count <= Self.maximumBytes else { throw ConversationFailure.tooLarge }
    try self.init(fields: JSONDecoder().decode(Object.self, from: data))
  }
  public init(fields: Object) throws {
    guard let id = fields["id"]?.string, Self.validID(id),
      let raw = fields["messages"]?.array
    else {
      throw ConversationFailure.invalid("Invalid conversation document")
    }
    var seen = Set<String>()
    for rawMessage in raw {
      guard let m = rawMessage.object, let id = m["id"]?.string, Self.validID(id),
        seen.insert(id).inserted, m["role"]?.string != nil,
        let content = m["content"]?.array, content.allSatisfy({ $0.object != nil }),
        m["parentId"] == nil || m["parentId"] == .null || m["parentId"]?.string != nil
      else { throw ConversationFailure.invalid("Invalid or duplicate message identity") }
    }
    self.fields = fields
  }
  public func encoded() throws -> Data {
    let data = try JSONEncoder().encode(fields)
    guard data.count <= Self.maximumBytes else { throw ConversationFailure.tooLarge }
    return data
  }
  public static func validID(_ id: String) -> Bool {
    !id.isEmpty && id.utf8.count <= 128
      && id.utf8.allSatisfy {
        (48...57).contains($0) || (65...90).contains($0)
          || (97...122).contains($0) || $0 == 45 || $0 == 95
      }
  }
  public mutating func rename(_ title: String) throws {
    let title = title.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !title.isEmpty, title.utf16.count <= 512,
      !title.unicodeScalars.contains(where: { $0.value < 32 })
    else {
      throw ConversationFailure.invalid("Enter a title on one line, up to 512 characters")
    }
    fields["title"] = .string(title)
    fields["titleSource"] = .string("manual")
    fields["titleModel"] = nil
    fields["titleCostUsd"] = nil
  }
  public mutating func setPinned(_ pinned: Bool) { fields["pinned"] = .bool(pinned) }
  mutating func replaceMessages(_ messages: [Object]) {
    fields["messages"] = .array(messages.map(Value.object))
  }
  mutating func setLeaf(_ id: String?) { fields["leafId"] = id.map(Value.string) }
  mutating func setMemory(_ memory: [String: String]) {
    fields["branchMemory"] = .object(memory.mapValues(Value.string))
  }
  public static func text(_ message: Object) -> String {
    (message["content"]?.array ?? []).compactMap {
      $0["type"]?.string == "text" ? $0["text"]?.string : nil
    }.joined(separator: "\n")
  }
}
