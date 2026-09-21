import Foundation

/// Incremental byte parser. UTF-8 is decoded only at line boundaries; a socket
/// chunk may split any scalar, CRLF, field or event. No per-token UI dispatch.
public struct ServerSentEvents: Sendable {
  public struct Event: Sendable, Equatable {
    public let name: String
    public let data: String
  }
  private var line = Data()
  private var data = [String]()
  private var dataBytes = 0
  private var eventName = "message"
  private var cr = false
  private var firstLine = true
  public let maximumBytes: Int
  public init(maximumBytes: Int = 4 * 1024 * 1024) { self.maximumBytes = max(1, maximumBytes) }

  public mutating func push(_ bytes: Data) throws -> [Event] {
    var result = [Event]()
    for byte in bytes { if let event = try push(byte) { result.append(event) } }
    return result
  }
  public mutating func push(_ byte: UInt8) throws -> Event? {
    if cr {
      cr = false
      if byte == 10 { return nil }
    }
    if byte == 13 || byte == 10 {
      cr = byte == 13
      return try finishLine()
    }
    guard line.count + dataBytes < maximumBytes else { throw ConversationFailure.tooLarge }
    line.append(byte)
    return nil
  }
  private mutating func finishLine() throws -> Event? {
    guard var text = String(data: line, encoding: .utf8) else {
      throw ConversationFailure.invalid("Invalid UTF-8 in response stream")
    }
    line.removeAll(keepingCapacity: true)
    if firstLine {
      firstLine = false
      if text.hasPrefix("\u{feff}") { text.removeFirst() }
    }
    if text.isEmpty {
      let value = data.isEmpty ? nil : Event(name: eventName, data: data.joined(separator: "\n"))
      data.removeAll(keepingCapacity: true)
      dataBytes = 0
      eventName = "message"
      return value
    }
    if text.hasPrefix(":") { return nil }
    let split = text.firstIndex(of: ":")
    let name = split.map { String(text[..<$0]) } ?? text
    var value = split.map { String(text[text.index(after: $0)...]) } ?? ""
    if value.hasPrefix(" ") { value.removeFirst() }
    if name == "data" {
      guard value.utf8.count + dataBytes + 1 <= maximumBytes else {
        throw ConversationFailure.tooLarge
      }
      data.append(value)
      dataBytes += value.utf8.count + 1
    } else if name == "event" {
      eventName = value
    }
    return nil
  }
  /// Never manufacture a completion from an unterminated event at EOF. The
  /// semantic decoder, not EOF or [DONE], decides whether a response completed.
  public mutating func finish() {
    line.removeAll()
    data.removeAll()
    dataBytes = 0
  }
}
