import Foundation

public enum LogCommand: Sendable, Encodable {
  case open(port: UInt16)
  case poll(id: String)
  case close(id: String)
  private enum CodingKeys: String, CodingKey { case kind, port, id }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .open(let port):
      try c.encode("open", forKey: .kind)
      try c.encode(port, forKey: .port)
    case .poll(let id):
      try c.encode("poll", forKey: .kind)
      try c.encode(id, forKey: .id)
    case .close(let id):
      try c.encode("close", forKey: .kind)
      try c.encode(id, forKey: .id)
    }
  }
}
public struct LogReply: Decodable, Sendable {
  public let id: String?
  public let state: String
  public let text: String
}
