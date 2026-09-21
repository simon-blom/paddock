import Foundation

/// Bounded, ordered, atomic presentation transfer. Partial states never reach
/// SwiftUI. A reload/new transfer discards an incomplete previous frame set.
public struct StudioPresentationFrames {
  private var transfer: Int?
  private var next = 0
  private var count = 0
  private var expected = 0
  private var buffer = Data()
  public init() {}
  public mutating func reset() { self = Self() }
  public mutating func accept(_ object: [String: Any]) throws -> Data? {
    guard let id = object["transfer"] as? Int, let index = object["index"] as? Int,
      let count = object["count"] as? Int, let bytes = object["bytes"] as? Int,
      let text = object["payload"] as? String, text.utf8.count <= 65536,
      let payload = Data(base64Encoded: text), payload.count <= 48 * 1024,
      bytes > 0, bytes <= 64 * 1024 * 1024,
      count == (bytes + 48 * 1024 - 1) / (48 * 1024), index >= 0, index < count
    else {
      reset()
      throw FrameError.invalid
    }
    if index == 0 {
      reset()
      transfer = id
      self.count = count
      expected = bytes
      buffer.reserveCapacity(bytes)
    }
    guard transfer == id, next == index, self.count == count, expected == bytes,
      buffer.count + payload.count <= expected
    else {
      reset()
      throw FrameError.invalid
    }
    buffer.append(payload)
    next += 1
    guard next == count else { return nil }
    guard buffer.count == expected else {
      reset()
      throw FrameError.invalid
    }
    let result = buffer
    reset()
    return result
  }
  public enum FrameError: Error { case invalid }
}
