import Darwin
import Foundation

public protocol ManagerLoading: Sendable {
  func prepareEndpoint(model: String, artifact: String) async throws -> ConfiguredEndpoint
  func snapshot() async throws -> ManagerSnapshot
  func logs(_ command: LogCommand) async throws -> LogReply
  func maintenance(_ command: MaintenanceCommand) async throws -> MaintenanceReply
  func submit(_ command: ModelCommand) async throws -> ManagementJob
  func chat(_ command: ChatCommand) async throws -> ChatReply
  func studio(assets: URL) async throws -> StudioHost
  func nativeConversationHost() async throws -> StudioHost
  func downloads(_ command: DownloadCommand) async throws -> DownloadReply
  func connections(_ command: ConnectionCommand) async throws -> ConnectionReply
  func integrations(_ command: IntegrationCommand) async throws -> IntegrationReply
  func close() async
}

extension ManagerLoading {
  public func maintenance(_ command: MaintenanceCommand) async throws -> MaintenanceReply {
    throw ManagerError.core("Management insights are unavailable from this source.")
  }
  public func prepareEndpoint(model: String, artifact: String) async throws -> ConfiguredEndpoint {
    throw ManagerError.core("Model settings are unavailable from this management source.")
  }
  public func logs(_ command: LogCommand) async throws -> LogReply {
    throw ManagerError.core("Logs are unavailable from this management source.")
  }
  public func integrations(_ command: IntegrationCommand) async throws -> IntegrationReply {
    throw ManagerError.core("Tools management is unavailable from this source.")
  }
  public func connections(_ command: ConnectionCommand) async throws -> ConnectionReply {
    throw ManagerError.core("Connection management is unavailable from this source.")
  }
  public func downloads(_ command: DownloadCommand) async throws -> DownloadReply {
    throw ManagerError.core("Download management is unavailable from this source.")
  }
  public func studio(assets: URL) async throws -> StudioHost {
    throw ManagerError.core("The bundled Studio host is unavailable.")
  }
  public func nativeConversationHost() async throws -> StudioHost {
    throw ManagerError.core("The native conversation service is unavailable.")
  }
  public func chat(_ command: ChatCommand) async throws -> ChatReply {
    throw ManagerError.core("Chat is unavailable from this management source.")
  }
  public func close() async {}
  public func submit(_ command: ModelCommand) async throws -> ManagementJob {
    throw ManagerError.core("This management source is read-only.")
  }

}

public enum ManagerError: Error, LocalizedError, Sendable, Equatable {
  case core(String)
  case incompatibleABI, wrongService
  case unsupportedCatalog(Int)
  case closed

  public var errorDescription: String? {
    switch self {
    case .core(let message): message
    case .incompatibleABI:
      "The bundled management core does not match this app. Rebuild or reinstall Paddock."
    case .wrongService: "The bundled core returned an invalid management identity."
    case .unsupportedCatalog(let schema):
      "This app cannot read catalog schema \(schema). Update Paddock."
    case .closed: "Paddock is shutting down."
    }
  }
}

/// The app owns the Rust core. Management calls use the private ABI; the full
/// Studio gets an app-private host, never a separate executable. Blocking ABI work and
/// JSON decoding run on a dedicated serial queue, not a Swift actor executor.
public final class NativeManager: ManagerLoading, Sendable {
  private let queue = DispatchQueue(label: "io.truespar.paddock.management", qos: .userInitiated)
  private let storage: CoreStorage

  public func maintenance(_ command: MaintenanceCommand) async throws -> MaintenanceReply {
    try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(with: Result { try storage.maintenance(command) })
      }
    }
  }

  public func prepareEndpoint(model: String, artifact: String) async throws -> ConfiguredEndpoint {
    try Task.checkCancellation()
    return try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(
          with: Result { try storage.modelRequest(.prepare(model: model, artifact: artifact)) })
      }
    }
  }

  public func logs(_ command: LogCommand) async throws -> LogReply {
    // Close must run even after a view task is cancelled, or its subscription
    // would linger until the core's idle deadline.
    try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(with: Result { try storage.logs(command) })
      }
    }
  }

  public func integrations(_ command: IntegrationCommand) async throws -> IntegrationReply {
    try Task.checkCancellation()
    return try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(with: Result { try storage.integrations(command) })
      }
    }
  }

  public func connections(_ command: ConnectionCommand) async throws -> ConnectionReply {
    try Task.checkCancellation()
    return try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(with: Result { try storage.connections(command) })
      }
    }
  }

  public func downloads(_ command: DownloadCommand) async throws -> DownloadReply {
    try Task.checkCancellation()
    return try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(with: Result { try storage.downloads(command) })
      }
    }
  }

  public convenience init() {
    self.init(
      libraryURL: Bundle.main.privateFrameworksURL?.appending(path: "libpaddock_desktop.dylib"))
  }

  // Tests supply an explicit build artifact. Production loads only the library
  // inside this app bundle, never PATH, an environment variable or a URL.
  init(libraryURL: URL?) { storage = CoreStorage(libraryURL: libraryURL) }

  deinit {
    let storage = storage
    queue.async { storage.close() }
  }

  public func snapshot() async throws -> ManagerSnapshot {
    try Task.checkCancellation()
    let value: ManagerSnapshot = try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(with: Result { try storage.snapshot() })
      }
    }
    try Task.checkCancellation()
    return value
  }

  public func close() async {
    await withCheckedContinuation { continuation in
      queue.async { [storage] in
        storage.close()
        continuation.resume()
      }
    }
  }

  public func chat(_ command: ChatCommand) async throws -> ChatReply {
    try Task.checkCancellation()
    return try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in continuation.resume(with: Result { try storage.chat(command) }) }
    }
    // An accepted send must deliver its receipt even if its caller cancelled.
    // Explicit cancel names that receipt; do not orphan a background generation.
  }

  public func studio(assets: URL) async throws -> StudioHost {
    try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in continuation.resume(with: Result { try storage.studio(assets) }) }
    }
  }

  public func nativeConversationHost() async throws -> StudioHost {
    try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in continuation.resume(with: Result { try storage.studio(nil) }) }
    }
  }

  public func submit(_ command: ModelCommand) async throws -> ManagementJob {
    try Task.checkCancellation()
    return try await withCheckedThrowingContinuation { continuation in
      queue.async { [storage] in
        continuation.resume(with: Result { try storage.submit(command) })
      }
    }
    // Once accepted, always deliver the receipt, even if the caller cancelled.
    // A cancelled Swift Task cannot silently cancel a Rust lifecycle operation.
  }
}

/// All mutable fields are accessed exclusively on NativeManager.queue. No raw
/// pointer crosses into a Task or UI object; close cannot race a Rust call.
private final class CoreStorage: @unchecked Sendable {
  let libraryURL: URL?
  private var library: CoreLibrary?
  private var core: UnsafeMutableRawPointer?
  private var isClosed = false

  init(libraryURL: URL?) { self.libraryURL = libraryURL }

  private func open() throws -> CoreLibrary {
    guard !isClosed else { throw ManagerError.closed }
    if library == nil { library = try CoreLibrary(url: libraryURL) }
    guard let library else { throw ManagerError.incompatibleABI }
    var error: UnsafeMutablePointer<CChar>?
    if core == nil {
      core = library.open(&error)
      guard core != nil else { throw library.consumeError(error) }
    }
    return library
  }

  func snapshot() throws -> ManagerSnapshot {
    let library = try open()
    var error: UnsafeMutablePointer<CChar>?
    guard let json = library.snapshot(core, &error) else { throw library.consumeError(error) }
    defer { library.free(json) }
    // Rust bounds the payload to 8 MiB, owns its allocation, and NUL-terminates
    // it. Swift makes its own copy before freeing on the same serial queue.
    let data = Data(bytes: json, count: strlen(json))
    let result = try ManagerWire.decode(ManagerSnapshot.self, from: data)
    guard result.identity.role == "manager" else { throw ManagerError.wrongService }
    guard result.catalog.schema == 3 else {
      throw ManagerError.unsupportedCatalog(result.catalog.schema)
    }
    return result
  }

  func submit(_ command: ModelCommand) throws -> ManagementJob {
    try modelRequest(command)
  }

  func modelRequest<T: Decodable>(_ command: ModelCommand) throws -> T {
    let library = try open()
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(command)
    var error: UnsafeMutablePointer<CChar>?
    let json = data.withUnsafeBytes { bytes in
      library.submit(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let json else { throw library.consumeError(error) }
    defer { library.free(json) }
    return try ManagerWire.decode(T.self, from: Data(bytes: json, count: strlen(json)))
  }

  func logs(_ command: LogCommand) throws -> LogReply {
    let library = try open()
    let data = try JSONEncoder().encode(command)
    var error: UnsafeMutablePointer<CChar>?
    let json = data.withUnsafeBytes { bytes in
      library.logs(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let json else { throw library.consumeError(error) }
    defer { library.free(json) }
    return try JSONDecoder().decode(LogReply.self, from: Data(bytes: json, count: strlen(json)))
  }

  func maintenance(_ command: MaintenanceCommand) throws -> MaintenanceReply {
    let library = try open()
    let data = try JSONEncoder().encode(command)
    var error: UnsafeMutablePointer<CChar>?
    let json = data.withUnsafeBytes { bytes in
      library.maintenance(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let json else { throw library.consumeError(error) }
    defer { library.free(json) }
    return try JSONDecoder().decode(
      MaintenanceReply.self, from: Data(bytes: json, count: strlen(json)))
  }

  func close() {
    if let core { library?.close(core) }
    core = nil
    isClosed = true
  }

  func downloads(_ command: DownloadCommand) throws -> DownloadReply {
    let library = try open()
    let data = try JSONEncoder().encode(command)
    var error: UnsafeMutablePointer<CChar>?
    let json = data.withUnsafeBytes { bytes in
      library.downloads(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let json else { throw library.consumeError(error) }
    defer { library.free(json) }
    return try ManagerWire.decode(DownloadReply.self, from: Data(bytes: json, count: strlen(json)))
  }

  func connections(_ command: ConnectionCommand) throws -> ConnectionReply {
    let library = try open()
    let data = try JSONEncoder().encode(command)
    var error: UnsafeMutablePointer<CChar>?
    let json = data.withUnsafeBytes { bytes in
      library.connections(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let json else { throw library.consumeError(error) }
    defer { library.free(json) }
    // Connection metadata deliberately uses the shared web's camelCase names.
    return try JSONDecoder().decode(
      ConnectionReply.self, from: Data(bytes: json, count: strlen(json)))
  }

  func integrations(_ command: IntegrationCommand) throws -> IntegrationReply {
    let library = try open()
    let data = try JSONEncoder().encode(command)
    var error: UnsafeMutablePointer<CChar>?
    let json = data.withUnsafeBytes { bytes in
      library.integrations(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let json else { throw library.consumeError(error) }
    defer { library.free(json) }
    return try JSONDecoder().decode(
      IntegrationReply.self, from: Data(bytes: json, count: strlen(json)))
  }

  func chat(_ command: ChatCommand) throws -> ChatReply {
    let library = try open()
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(command)
    var error: UnsafeMutablePointer<CChar>?
    let result = data.withUnsafeBytes { bytes in
      library.chat(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let result else { throw library.consumeError(error) }
    defer { library.free(result) }
    return try ManagerWire.decode(ChatReply.self, from: Data(bytes: result, count: strlen(result)))
  }

  func studio(_ assets: URL?) throws -> StudioHost {
    guard assets == nil || assets?.isFileURL == true else {
      throw ManagerError.core("Studio assets must be bundled locally.")
    }
    let library = try open()
    let data = try JSONEncoder().encode(assets.map { ["assets": $0.path] } ?? [:])
    var error: UnsafeMutablePointer<CChar>?
    let result = data.withUnsafeBytes { bytes in
      library.studio(
        core, bytes.baseAddress?.assumingMemoryBound(to: UInt8.self), bytes.count, &error)
    }
    guard let result else { throw library.consumeError(error) }
    defer { library.free(result) }
    return try JSONDecoder().decode(
      StudioHost.self, from: Data(bytes: result, count: strlen(result)))
  }
}

/// Signatures mirror crates/paddock-desktop/include/paddock_desktop.h.
/// ABI version is checked before opening any product state. Dynamic loading
/// keeps pure Swift tests independent of a Rust build; distribution bundles
/// and signs this library as part of the single application.
struct CoreLibrary {
  typealias Version = @convention(c) () -> UInt32
  typealias Open =
    @convention(c) (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?) -> UnsafeMutableRawPointer?
  typealias Snapshot =
    @convention(c) (UnsafeMutableRawPointer?, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?)
    -> UnsafeMutablePointer<CChar>?
  typealias Free = @convention(c) (UnsafeMutablePointer<CChar>?) -> Void
  typealias Submit =
    @convention(c) (
      UnsafeMutableRawPointer?, UnsafePointer<UInt8>?, Int,
      UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> UnsafeMutablePointer<CChar>?
  typealias Close = @convention(c) (UnsafeMutableRawPointer?) -> Void
  typealias Browse =
    @convention(c) (UnsafePointer<UInt8>?, Int, UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?)
    -> UnsafeMutablePointer<CChar>?
  let open: Open
  let snapshot: Snapshot
  let submit: Submit
  let free: Free
  let close: Close
  let browse: Browse
  let chat: Submit
  let studio: Submit
  let downloads: Submit
  let connections: Submit
  let integrations: Submit
  let logs: Submit
  let maintenance: Submit

  init(url: URL?) throws {
    guard let url, url.isFileURL, FileManager.default.fileExists(atPath: url.path) else {
      throw ManagerError.core(
        "The embedded management core is missing. Build or reinstall the complete Paddock.app bundle."
      )
    }
    guard let handle = dlopen(url.path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE) else {
      throw ManagerError.core(
        "The embedded management core could not load. Rebuild or reinstall Paddock.")
    }
    // NODELETE is deliberate: Rust/Tokio thread-local destructors must never
    // jump into an unloaded dylib. Core resources still close explicitly.
    defer { dlclose(handle) }
    func symbol<T>(_ name: String, as type: T.Type) throws -> T {
      guard let pointer = dlsym(handle, name) else { throw ManagerError.incompatibleABI }
      return unsafeBitCast(pointer, to: type)
    }
    let version = try symbol("paddock_desktop_abi_version", as: Version.self)
    guard version() == 12 else { throw ManagerError.incompatibleABI }
    open = try symbol("paddock_desktop_open", as: Open.self)
    snapshot = try symbol("paddock_desktop_snapshot", as: Snapshot.self)
    submit = try symbol("paddock_desktop_submit", as: Submit.self)
    free = try symbol("paddock_desktop_string_free", as: Free.self)
    close = try symbol("paddock_desktop_close", as: Close.self)
    browse = try symbol("paddock_desktop_browse", as: Browse.self)
    chat = try symbol("paddock_desktop_chat", as: Submit.self)
    studio = try symbol("paddock_desktop_studio", as: Submit.self)
    downloads = try symbol("paddock_desktop_downloads", as: Submit.self)
    connections = try symbol("paddock_desktop_connections", as: Submit.self)
    integrations = try symbol("paddock_desktop_integrations", as: Submit.self)
    logs = try symbol("paddock_desktop_logs", as: Submit.self)
    maintenance = try symbol("paddock_desktop_maintenance", as: Submit.self)
  }

  func consumeError(_ pointer: UnsafeMutablePointer<CChar>?) -> ManagerError {
    guard let pointer else {
      return .core("The embedded management core failed without an error description.")
    }
    defer { free(pointer) }
    return .core(String(cString: pointer))
  }
}
