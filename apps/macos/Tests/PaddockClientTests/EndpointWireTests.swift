import Foundation
import Testing

@testable import PaddockClient

@Suite("Real native endpoint edit boundary", .serialized)
struct EndpointWireTests {
  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_DESKTOP_TEST_LIBRARY"] != nil
        && URL(fileURLWithPath: ProcessInfo.processInfo.environment["PADDOCK_DATA"] ?? "/")
          .lastPathComponent.hasPrefix("paddock-endpoint-test.")
    ))
  func savedConfigAndJobReceiptsRoundTripWithoutRunningAModel() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let library = env["PADDOCK_DESKTOP_TEST_LIBRARY"],
      let root = env["PADDOCK_DATA"],
      URL(fileURLWithPath: root).lastPathComponent.hasPrefix("paddock-endpoint-test.")
    else { return }
    let servers = URL(fileURLWithPath: root).appending(path: "servers")
    let file = servers.appending(path: "13495.toml")
    #expect(!FileManager.default.fileExists(atPath: file.path))
    guard !FileManager.default.fileExists(atPath: file.path) else { return }
    try FileManager.default.createDirectory(at: servers, withIntermediateDirectories: true)
    let original =
      "# wire fixture\nport = 13495\nhost = '127.0.0.1'\ndevice = 'metal'\nmodel = 'fixture.gguf'\nmax_ctx = 4096\napi_key = 'synthetic-wire-secret'\ncustom_value = true\n"
    try Data(original.utf8).write(to: file, options: .atomic)
    let client = NativeManager(libraryURL: URL(fileURLWithPath: library))
    let before = try #require(
      try await client.snapshot().servers?.first(where: { $0.port == 13495 }))
    #expect(before.settings?.hasApiKey == true && before.settings?.maxCtx == 4096)
    #expect(before.settings?.runtimeOptions?.count == 16)
    let revision = try #require(before.revision)
    var receipt = try await client.submit(
      .edit(
        port: 13495, revision: revision, pid: nil,
        changes: [
          .maxCtx(8192),
          .runtime([
            "temp": .number(0.25), "max_tokens": .integer(2048), "no_metrics": .boolean(true),
          ]), .kvOffload(.init(enabled: false, ramGb: 1, nvmeGb: 8)),
        ], apply: .defer, allowNetwork: false))
    let ticket = receipt.id
    for _ in 0..<300 {
      if !receipt.isActive { break }
      try await Task.sleep(for: .milliseconds(10))
      receipt = try await client.submit(.poll(id: ticket))
    }
    #expect(receipt.state == "succeeded")
    let snapshot = try await client.snapshot()
    #expect(!snapshot.runners.contains(where: { $0.port == 13495 }))
    let after = try #require(snapshot.servers?.first(where: { $0.port == 13495 }))
    #expect(after.settings?.maxCtx == 8192 && after.revision != revision)
    #expect(after.settings?.kvOffload == .init(enabled: false, ramGb: 1, nvmeGb: 8))
    #expect(after.settings?.runtimeOptions?.first { $0.id == "temp" }?.value == .number(0.25))
    #expect(
      after.settings?.runtimeOptions?.first { $0.id == "max_tokens" }?.value == .integer(2048))
    #expect(
      after.settings?.runtimeOptions?.first { $0.id == "no_metrics" }?.value == .boolean(true))
    let saved = try String(contentsOf: file, encoding: .utf8)
    #expect(saved.contains("synthetic-wire-secret") && saved.contains("custom_value = true"))
    var stale = try await client.submit(
      .edit(
        port: 13495, revision: revision, pid: nil,
        changes: [.maxCtx(16384)], apply: .defer, allowNetwork: false))
    for _ in 0..<300 {
      if !stale.isActive { break }
      try await Task.sleep(for: .milliseconds(10))
      stale = try await client.submit(.poll(id: stale.id))
    }
    #expect(stale.state == "failed" && stale.message.contains("changed"))
    #expect(try String(contentsOf: file, encoding: .utf8) == saved)
    #expect(!stale.message.contains("synthetic-wire-secret"))
    await client.close()
  }
}
