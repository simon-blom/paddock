import Darwin
import Foundation

/// The web manager's normalized public projection. Prices are USD per token,
/// not per million; nil is unknown and must never be interpreted as free.
public struct CloudModel: Decodable, Sendable, Identifiable, Equatable {
  public let id: String
  public let display: String?
  public let ctx: UInt64?
  public let maxOut: UInt64?
  public let promptPrice: Double?
  public let completionPrice: Double?
  public let vision: Bool?
  public let reasoning: Bool?
  public let tools: Bool?
  public let asr: Bool?
  public let free: Bool?
  public let created: UInt64?
  public let blurb: String?
}

public struct CloudCatalog: Decodable, Sendable {
  public let models: [CloudModel]
  public let ranked: Bool
}

public struct CloudProvider: Decodable, Sendable, Equatable {
  public let name: String
  public let tag: String?
  public let ctx: UInt64?
  public let maxOut: UInt64?
  public let promptPrice: Double?
  public let completionPrice: Double?
  public let quant: String?
  public let tps: Double?

  /// Same pin as the web Studio: tag distinguishes region/tier variants.
  public var routingID: String { tag.flatMap { $0.isEmpty ? nil : $0 } ?? name }
}

/// A pending Studio pick, not a saved account or an active request. Mirrors the
/// web's stable pick contract; live prices never become persisted billing facts.
public struct CloudModelPick: Codable, Sendable, Equatable {
  public let id: String
  public let display: String?
  public let ctx: UInt64?
  public let maxOut: UInt64?
  public let vision: Bool?
  public let reasoning: Bool?
  public let asr: Bool?
  public let provider: String?

  public init(model: CloudModel, provider: CloudProvider?) {
    id = model.id
    display = model.display
    ctx = provider?.ctx ?? model.ctx
    maxOut = provider?.maxOut ?? model.maxOut
    vision = model.vision
    reasoning = model.reasoning
    asr = model.asr
    self.provider = provider?.routingID
  }

  public var pickKey: String { provider.map { "\(id)@\($0)" } ?? id }

  /// Explicit manual-ID pick, like the web browser's no-results action. Unknown
  /// capabilities remain unknown; adding an ID never invents model entitlement.
  public init(id: String) {
    self.id = id
    display = nil
    ctx = nil
    maxOut = nil
    vision = nil
    reasoning = nil
    asr = nil
    provider = nil
  }
}

public struct CloudProviders: Decodable, Sendable {
  public let providers: [CloudProvider]
}

public protocol OpenRouterLoading: Sendable {
  func catalog() async throws -> CloudCatalog
  func providers(for model: String) async throws -> CloudProviders
}

/// A stateless public-catalog client. Separate workers prevent an internet
/// timeout from queuing behind (or blocking) local model management operations.
/// It neither opens a database nor accepts credentials or arbitrary URLs.
public final class NativeOpenRouter: OpenRouterLoading, Sendable {
  private let queue = DispatchQueue(
    label: "io.truespar.paddock.catalog", qos: .userInitiated, attributes: .concurrent)
  private let libraryURL: URL?

  public convenience init() {
    self.init(
      libraryURL: Bundle.main.privateFrameworksURL?.appending(path: "libpaddock_desktop.dylib"))
  }
  init(libraryURL: URL?) { self.libraryURL = libraryURL }

  public func catalog() async throws -> CloudCatalog {
    try await query(["kind": "catalog"], as: CloudCatalog.self)
  }
  public func providers(for model: String) async throws -> CloudProviders {
    try await query(["kind": "providers", "model": model], as: CloudProviders.self)
  }
  private func query<T: Decodable & Sendable>(_ request: [String: String], as type: T.Type)
    async throws -> T
  {
    try Task.checkCancellation()
    let data = try JSONEncoder().encode(request)
    let result: T = try await withCheckedThrowingContinuation { continuation in
      queue.async { [libraryURL] in
        continuation.resume(
          with: Result {
            let library = try CoreLibrary(url: libraryURL)
            var error: UnsafeMutablePointer<CChar>?
            let json = data.withUnsafeBytes { bytes in
              library.browse(
                bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
            }
            guard let json else { throw library.consumeError(error) }
            defer { library.free(json) }
            return try ManagerWire.decode(T.self, from: Data(bytes: json, count: strlen(json)))
          })
      }
    }
    try Task.checkCancellation()
    return result
  }
}
