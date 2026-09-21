import Foundation

public struct NativeConnector: Decodable, Sendable, Identifiable, Equatable {
  public let id: String
  public let label: String
  public let url: String
  public let registryKey: String
  public let system: Bool
  public let ports: [UInt16]
  public let revision: UInt64
  public let hasHeaders: Bool
  public let connected: Bool
  public let keychain: Bool
  public let oauthRevision: UInt64
  public let credentialReady: Bool?
}
public struct ConnectorHit: Decodable, Sendable, Identifiable {
  public struct Endpoint: Decodable, Sendable {
    public let url: String
    public let transport: String
  }
  public let key: String
  public let name: String
  public let description: String
  public let domain: String
  public let authorityTier: String
  public let liveness: String
  public let githubStars: UInt64?
  public let toolCount: UInt64?
  public let remoteEndpoints: [Endpoint]
  public let categories: [String]?
  public let spdxLicense: String?
  public let tools: [String]?
  public let repoUrl: String?
  public let homepage: String?
  public let lastHandshakeAt: String?
  public struct Connection: Decodable, Sendable {
    public let recommendedURL: String?
    public let authRequired: Bool?
    public let note: String?
  }
  public let connection: Connection?
  public var id: String { key }
}
public struct ConnectorTool: Decodable, Sendable, Identifiable {
  public let name: String
  public let description: String?
  public var id: String { name }
}
public struct SearchConfiguration: Decodable, Sendable {
  public let port: UInt16
  public let revision: String
  public let provider: String
  public let hasKey: Bool
  public init(port: UInt16, revision: String, provider: String, hasKey: Bool) {
    self.port = port
    self.revision = revision
    self.provider = provider
    self.hasKey = hasKey
  }
}
public struct ConnectorAuthorization: Decodable, Sendable, Identifiable {
  public let connectorId: String
  public let url: String
  public let revision: UInt64
  public var id: String { connectorId }
}
public struct ConnectorDraft: Encodable, Sendable, Equatable {
  public var id: String?
  public var revision: UInt64?
  public var label: String
  public var url: String
  /// nil preserves the saved headers; [:] explicitly removes them.
  public var headers: [String: String]?
  public var registryKey: String
  public init(row: NativeConnector? = nil) {
    id = row?.id
    revision = row?.revision
    label = row?.label ?? ""
    url = row?.url ?? ""
    registryKey = row?.registryKey ?? ""
    headers = row == nil ? [:] : nil
  }
}
public struct IntegrationValue: Decodable, Sendable {
  public let authRequired: Bool?
  public let connectors: [NativeConnector]?
  public let results: [ConnectorHit]?
  public let detail: ConnectorHit?
  public let tools: [ConnectorTool]?
  public let search: SearchConfiguration?
  public let savedId: String?
  public let message: String?
  public let authorization: ConnectorAuthorization?
}
public struct IntegrationJob: Decodable, Sendable {
  public let id: String
  public let status: String
  public let message: String
  public let value: IntegrationValue
}
public struct IntegrationReply: Decodable, Sendable { public let job: IntegrationJob? }

public enum IntegrationOperation: Encodable, Sendable {
  case list
  case search(String)
  case detail(String)
  case check(ConnectorDraft)
  case save(ConnectorDraft)
  case remove(id: String, revision: UInt64)
  case scope(id: String, revision: UInt64, all: Bool, ports: [UInt16])
  case tools(String)
  case searchSettings(UInt16)
  case saveSearch(port: UInt16, revision: String, provider: String, key: String?)
  case signIn(id: String, revision: UInt64, clientID: String?)
  case cancelSignIn(String)
  case disconnect(id: String, revision: UInt64)
  case unlock(id: String, revision: UInt64)
  public var mutation: Bool {
    switch self {
    case .save, .remove, .scope, .saveSearch, .signIn, .cancelSignIn, .disconnect, .unlock: true
    default: false
    }
  }
  private enum Keys: String, CodingKey {
    case kind, query, key, draft, id, revision, all, ports, port, provider
    case clientID = "client_id"
  }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.container(keyedBy: Keys.self)
    switch self {
    case .list: try c.encode("list", forKey: .kind)
    case .search(let query):
      try c.encode("search", forKey: .kind)
      try c.encode(query, forKey: .query)
    case .detail(let key):
      try c.encode("detail", forKey: .kind)
      try c.encode(key, forKey: .key)
    case .check(let draft), .save(let draft):
      try c.encode(mutation ? "save" : "check", forKey: .kind)
      try c.encode(draft, forKey: .draft)
    case .remove(let id, let revision):
      try c.encode("remove", forKey: .kind)
      try c.encode(id, forKey: .id)
      try c.encode(revision, forKey: .revision)
    case .scope(let id, let revision, let all, let ports):
      try c.encode("scope", forKey: .kind)
      try c.encode(id, forKey: .id)
      try c.encode(revision, forKey: .revision)
      try c.encode(all, forKey: .all)
      try c.encode(ports, forKey: .ports)
    case .tools(let id):
      try c.encode("tools", forKey: .kind)
      try c.encode(id, forKey: .id)
    case .searchSettings(let port):
      try c.encode("search_settings", forKey: .kind)
      try c.encode(port, forKey: .port)
    case .saveSearch(let port, let revision, let provider, let key):
      try c.encode("save_search", forKey: .kind)
      try c.encode(port, forKey: .port)
      try c.encode(revision, forKey: .revision)
      try c.encode(provider, forKey: .provider)
      try c.encodeIfPresent(key, forKey: .key)
    case .signIn(let id, let revision, let clientID):
      try c.encode("sign_in", forKey: .kind)
      try c.encode(id, forKey: .id)
      try c.encode(revision, forKey: .revision)
      try c.encodeIfPresent(clientID, forKey: .clientID)
    case .cancelSignIn(let id):
      try c.encode("cancel_sign_in", forKey: .kind)
      try c.encode(id, forKey: .id)
    case .disconnect(let id, let revision):
      try c.encode("disconnect", forKey: .kind)
      try c.encode(id, forKey: .id)
      try c.encode(revision, forKey: .revision)
    case .unlock(let id, let revision):
      try c.encode("unlock", forKey: .kind)
      try c.encode(id, forKey: .id)
      try c.encode(revision, forKey: .revision)
    }
  }
}
public enum IntegrationCommand: Encodable, Sendable {
  case run(IntegrationOperation)
  case poll(String)
  case cancel(String)
  private enum Keys: String, CodingKey { case kind, operation, id }
  public func encode(to encoder: any Encoder) throws {
    var c = encoder.container(keyedBy: Keys.self)
    switch self {
    case .run(let op):
      try c.encode("run", forKey: .kind)
      try c.encode(op, forKey: .operation)
    case .poll(let id):
      try c.encode("poll", forKey: .kind)
      try c.encode(id, forKey: .id)
    case .cancel(let id):
      try c.encode("cancel", forKey: .kind)
      try c.encode(id, forKey: .id)
    }
  }
}

extension ManagerLoading {
  /// Callers use an app-owned task for mutations. Cancelling queries also
  /// cancels the Rust network operation; a UI change cannot abort a save.
  public func integration(_ operation: IntegrationOperation) async throws -> IntegrationValue {
    guard var job = try await integrations(.run(operation)).job else {
      throw ManagerError.core("Missing tools receipt.")
    }
    let id = job.id
    do {
      while job.status == "running" {
        try await Task.sleep(for: .milliseconds(150))
        guard let next = try await integrations(.poll(id)).job else {
          throw ManagerError.core("Missing tools result.")
        }
        job = next
      }
      guard job.status == "succeeded" else { throw ManagerError.core(job.message) }
      _ = try? await integrations(.cancel(id))
      return job.value
    } catch {
      if !operation.mutation {
        _ = await Task { try? await self.integrations(.cancel(id)) }.value
      }
      throw error
    }
  }
}
