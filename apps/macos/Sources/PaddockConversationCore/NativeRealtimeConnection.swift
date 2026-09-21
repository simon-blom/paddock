import Foundation

/// One authenticated native socket. No cookies or keys leave the transport.
public actor NativeRealtimeConnection {
  private let task: URLSessionWebSocketTask
  init(task: URLSessionWebSocketTask) { self.task = task }
  public func send(_ value: [String: ConversationValue]) async throws {
    try Task.checkCancellation()
    let data = try JSONEncoder().encode(value)
    guard data.count <= 2 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
    try await task.send(.string(String(decoding: data, as: UTF8.self)))
  }
  public func receive() async throws -> [String: ConversationValue] {
    let message = try await task.receive()
    let data: Data
    switch message {
    case .string(let text): data = Data(text.utf8)
    case .data(let bytes): data = bytes
    @unknown default: throw ConversationFailure.invalid("Invalid speech message")
    }
    guard data.count <= 4 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
    return try JSONDecoder().decode([String: ConversationValue].self, from: data)
  }
  public func close() { task.cancel(with: .goingAway, reason: nil) }
  deinit { task.cancel(with: .goingAway, reason: nil) }
}
