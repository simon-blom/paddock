import Foundation
import Testing

@testable import PaddockClient

@Suite("Real Rust router contract")
struct RouterContractTests {
  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Run apps/macos/scripts/check.sh to generate the real manager responses first."))
  func decodesCurrentManagerResponses() throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"])
    let directory = URL(fileURLWithPath: path, isDirectory: true)
    let identity = try ManagerWire.decode(
      ManagerIdentity.self, from: Data(contentsOf: directory.appending(path: "server.json")))
    let readiness = try ManagerWire.decode(
      Readiness.self, from: Data(contentsOf: directory.appending(path: "readiness.json")))
    let catalog = try ManagerWire.decode(
      ModelCatalog.self, from: Data(contentsOf: directory.appending(path: "catalog.json")))
    let runners = try ManagerWire.decode(
      [RunnerInfo].self, from: Data(contentsOf: directory.appending(path: "runners.json")))
    #expect(identity.role == "manager")
    #expect(readiness.backend == "metal")
    #expect(catalog.schema == 3)
    #expect(!catalog.models.isEmpty)
    #expect(runners.isEmpty)
    let qwen = try #require(catalog.models.first { $0.id == "qwen3.8-27b" })
    let mlx = try #require(qwen.artifacts.first { $0.id == "mlx-4bit" })
    #expect(mlx.supports(backend: readiness.backend))
    #expect(mlx.source?.repo == "mlx-community/Qwen3.8-27B-4bit")
    #expect(!mlx.installed)
    let gemma = try #require(catalog.models.first { $0.id == "gemma-4-31b" })
    let vision = try #require(gemma.artifacts.first { $0.id == "mlx-4bit" })
    #expect(vision.runtime?.embeddedVision == true)
    #expect(vision.supports(backend: readiness.backend))
    #expect(vision.supportNotice == nil)
  }
}
