import Foundation
import Testing

@testable import PaddockClient

@Suite("Real native log subscription", .serialized)
struct NativeLogWireTests {
  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_DESKTOP_TEST_LIBRARY"] != nil
        && URL(fileURLWithPath: ProcessInfo.processInfo.environment["PADDOCK_DATA"] ?? "/")
          .lastPathComponent.hasPrefix("paddock-log-test.")
    ))
  func followsAppendsAndRotationWithoutLeakingCredentialsOrStartingModels() async throws {
    let env = ProcessInfo.processInfo.environment
    let root = URL(fileURLWithPath: try #require(env["PADDOCK_DATA"]))
    let logs = root.appending(path: "logs")
    let servers = root.appending(path: "servers")
    try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: servers, withIntermediateDirectories: true)
    let file = logs.appending(path: "runner-13495.log")
    #expect(!FileManager.default.fileExists(atPath: file.path))
    guard !FileManager.default.fileExists(atPath: file.path) else { return }
    try Data(
      "port=13495\nhost='127.0.0.1'\nmodel='fixture.gguf'\napi_key='synthetic-log-secret'\n".utf8
    )
    .write(to: servers.appending(path: "13495.toml"), options: .atomic)
    try Data("2026-09-16T09:12:01Z INFO runner: ready\necho synthetic-log-secret\n".utf8).write(
      to: file)
    let client = NativeManager(
      libraryURL: URL(fileURLWithPath: try #require(env["PADDOCK_DESKTOP_TEST_LIBRARY"])))
    let opened = try await client.logs(.open(port: 13495))
    let id = try #require(opened.id)
    var received = ""
    for _ in 0..<100 {
      received += try await client.logs(.poll(id: id)).text
      if received.contains("withheld") { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(received.contains("ready") && received.contains("withheld"))
    #expect(!received.contains("synthetic-log-secret"))
    #expect(try await client.logs(.poll(id: id)).text.isEmpty)
    let handle = try FileHandle(forWritingTo: file)
    try handle.seekToEnd()
    try handle.write(contentsOf: Data("continued €\n".utf8))
    try handle.close()
    var appended = ""
    for _ in 0..<100 {
      appended += try await client.logs(.poll(id: id)).text
      if appended.contains("continued €") { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(appended == "continued €\n")
    try FileManager.default.moveItem(at: file, to: logs.appending(path: "runner-13495.prev.log"))
    try Data(("new generation\n" + String(repeating: "larger replacement\n", count: 20)).utf8)
      .write(to: file)
    var rotated = ""
    for _ in 0..<100 {
      rotated += try await client.logs(.poll(id: id)).text
      if rotated.contains("new generation") { break }
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(rotated.contains("rotated") && rotated.contains("new generation"))
    _ = try await client.logs(.close(id: id))
    await #expect(throws: ManagerError.self) { try await client.logs(.poll(id: id)) }
    #expect(try await client.snapshot().runners.allSatisfy { $0.port != 13495 })
    await client.close()
  }
}
