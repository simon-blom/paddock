import Foundation

/// Native presentation projections of the existing Studio conversation format.
/// Swift never writes these back: Rust preserves unknown fields and owns saves.
public struct ChatDocument: Codable, Sendable, Identifiable, Equatable {
  public let id: String
  public var title: String
  public var model: String
  public var messages: [ChatMessage]
}
public struct ChatMessage: Codable, Sendable, Identifiable, Equatable {
  public struct Part: Codable, Sendable, Equatable {
    public let type: String
    public var text: String?
    public init(type: String, text: String?) {
      self.type = type
      self.text = text
    }
  }
  public let id: String
  public let role: String
  public var content: [Part]
  public var reasoning: String?
  public var streaming: Bool?
  public var stopped: Bool?
  public var error: String?
  public var incomplete: String?
  public var nativeStatus: String?
  public var model: String?
  public var usage: ChatUsage?
  public var nativeTextParts: [String: String]?
  public var nativeReasoningParts: [String: String]?
  public var text: String { content.compactMap(\.text).joined(separator: "\n") }
}
public struct ChatUsage: Codable, Sendable, Equatable {
  public let promptTokens: Int?
  public let completionTokens: Int?
  public let reasoningTokens: Int?
}
public struct ChatSummary: Decodable, Sendable, Identifiable {
  public let id: String
  public let title: String
  public let model: String
  public let updatedAt: Double
}
public struct ChatDelta: Codable, Sendable {
  public let kind: String
  public let outputIndex: Int
  public let contentIndex: Int
  public let delta: String
}
public struct ChatCompletion: Decodable, Sendable {
  public let conversationId: String
  public let status: String
  public let error: String?
  public let saveError: String?
}
public struct ChatReply: Decodable, Sendable {
  public let conversation: ChatDocument?
  public let conversations: [ChatSummary]?
  public let streamId: String?
  public let events: [ChatDelta]?
  public let done: ChatCompletion?
  public let accepted: Bool?
}

public struct ChatCommand: Encodable, Sendable {
  let kind: String
  let conversationId: String?
  let port: UInt16?
  let pid: UInt32?
  let text: String?
  let streamId: String?
  private init(
    _ kind: String, conversationId: String? = nil, port: UInt16? = nil,
    pid: UInt32? = nil, text: String? = nil, streamId: String? = nil
  ) {
    self.kind = kind
    self.conversationId = conversationId
    self.port = port
    self.pid = pid
    self.text = text
    self.streamId = streamId
  }
  public static var list: Self { Self("list") }
  public static func load(_ id: String) -> Self { Self("load", conversationId: id) }
  public static func send(conversationId: String?, runner: RunnerInfo, text: String) -> Self {
    Self("send", conversationId: conversationId, port: runner.port, pid: runner.pid, text: text)
  }
  public static func poll(_ id: String) -> Self { Self("poll", streamId: id) }
  public static func cancel(_ id: String) -> Self { Self("cancel", streamId: id) }
}
