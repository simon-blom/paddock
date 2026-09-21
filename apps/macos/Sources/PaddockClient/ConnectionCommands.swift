import Foundation

/// Public metadata only. Credentials are accepted once by the private native
/// ABI; no command retrieves them and no key enters the WebKit content bridge.
public struct CloudConnection: Decodable, Identifiable, Sendable, Equatable {
  public let id: String
  public let name: String
  public let kind: String
  public let baseUrl: String
  public let hasKey: Bool
  public let credentialReady: Bool?
  public let revision: UInt64
  public let allowUnauthenticated: Bool
  public let credentialStorage: String
  public let models: [CloudModelPick]
  public var isOpenRouter: Bool { baseUrl == ConnectionDraft.openRouterBase }
}

public struct ConnectionDraft: Encodable, Sendable, Equatable {
  public static let openRouterBase = "https://openrouter.ai/api/v1"
  public var id: String?
  public var revision: UInt64?
  public var name: String
  public var kind: String
  public var baseUrl: String
  public var apiKey: String?
  public var allowUnauthenticated: Bool
  public init(connection: CloudConnection? = nil, openRouter: Bool = false) {
    id = connection?.id
    revision = connection?.revision
    name = connection?.name ?? (openRouter ? "OpenRouter" : "")
    kind = connection?.kind ?? "openai-compat"
    baseUrl = connection?.baseUrl ?? (openRouter ? Self.openRouterBase : "")
    apiKey = nil
    allowUnauthenticated = connection?.allowUnauthenticated ?? false
  }
}

public struct ConnectionJob: Decodable, Sendable {
  public let id: String
  public let status: String
  public let message: String
  public let models: [CloudModel]
  public let endpoint: CloudConnection?
  public var active: Bool { status == "checking" || status == "saving" }
}
public struct ConnectionReply: Decodable, Sendable {
  public let connections: [CloudConnection]?
  public let job: ConnectionJob?
}
public enum ConnectionCommand: Sendable, Encodable {
  case unlock(id: String, revision: UInt64)
  case list
  case check(ConnectionDraft)
  case poll(String)
  case cancel(String)
  case save(receipt: String, models: [CloudModelPick])
  case models(id: String, revision: UInt64, models: [CloudModelPick])
  case remove(id: String, revision: UInt64)
  private enum CodingKeys: String, CodingKey { case kind, id, draft, revision, models }
  public func encode(to encoder: any Encoder) throws {
    var values = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .unlock(let id, let revision):
      try values.encode("unlock", forKey: .kind)
      try values.encode(id, forKey: .id)
      try values.encode(revision, forKey: .revision)
    case .list: try values.encode("list", forKey: .kind)
    case .check(let draft):
      try values.encode("check", forKey: .kind)
      try values.encode(draft, forKey: .draft)
    case .poll(let id), .cancel(let id):
      if case .poll = self {
        try values.encode("poll", forKey: .kind)
      } else {
        try values.encode("cancel", forKey: .kind)
      }
      try values.encode(id, forKey: .id)
    case .save(let id, let models):
      try values.encode("save", forKey: .kind)
      try values.encode(id, forKey: .id)
      try values.encode(models, forKey: .models)
    case .models(let id, let revision, let models):
      try values.encode("models", forKey: .kind)
      try values.encode(id, forKey: .id)
      try values.encode(revision, forKey: .revision)
      try values.encode(models, forKey: .models)
    case .remove(let id, let revision):
      try values.encode("remove", forKey: .kind)
      try values.encode(id, forKey: .id)
      try values.encode(revision, forKey: .revision)
    }
  }
}
