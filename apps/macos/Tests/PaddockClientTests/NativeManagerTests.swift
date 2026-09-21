import Foundation
import Testing

@testable import PaddockClient

@Suite("Embedded core ABI", .timeLimit(.minutes(1)))
struct NativeManagerTests {
  @Test func missingCoreFailsClearlyWithoutAnExternalManagerFallback() async {
    let manager = NativeManager(libraryURL: nil)
    await #expect(throws: ManagerError.self) { try await manager.snapshot() }
    await manager.close()
  }

  @Test func closedCoreCannotReopen() async {
    let manager = NativeManager(libraryURL: nil)
    await manager.close()
    await manager.close()
    await #expect(throws: ManagerError.closed) { try await manager.snapshot() }
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_DESKTOP_TEST_LIBRARY"] != nil))
  func realRustCoreOwnsDirectoryAndReleasesItOnClose() async throws {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_DESKTOP_TEST_LIBRARY"])
    let dataPath = try #require(ProcessInfo.processInfo.environment["PADDOCK_DATA"])
    // The check script creates this isolated root; never open the user's DB.
    #expect(URL(fileURLWithPath: dataPath).lastPathComponent.hasPrefix("paddock-macos-test."))
    guard URL(fileURLWithPath: dataPath).lastPathComponent.hasPrefix("paddock-macos-test.") else {
      return
    }
    let url = URL(fileURLWithPath: path)
    let first = NativeManager(libraryURL: url)
    let second = NativeManager(libraryURL: url)
    do {
      let snapshot = try await first.snapshot()
      #expect(snapshot.identity.role == "manager")
      #expect(snapshot.readiness.backend == "metal")
      #expect(snapshot.catalog.schema == 3)
      #expect(snapshot.catalog.models.count > 10)
      #expect(try await first.downloads(.list).jobs.isEmpty)
      let download = try await first.downloads(.plan(model: "qwen3.8-27b", artifact: "mlx-4bit"))
      #expect(download.plan?.selection == ["mlx-4bit", "drafter2"])
      #expect((download.plan?.fileCount ?? 0) > 5)
      #expect(download.jobs.isEmpty)
      // Discovery can report an already-running endpoint even with an empty
      // private data root. Assert no saved endpoints/jobs were created here;
      // requiring an empty network fleet makes this test depend on other apps.
      #expect(snapshot.servers?.isEmpty != false)
      #expect(snapshot.jobs?.isEmpty != false)
      #expect(snapshot.identity.registry?.modelsDir == dataPath + "/models")
      await #expect(throws: ManagerError.self) {
        try await first.submit(.stop(port: 80, pid: 42))
      }
      let receipt = try await first.submit(
        .create(CreateEndpointRequest(model: "not-a-model", artifact: "mlx", port: 12345)))
      #expect(receipt.isActive)
      var completed: ManagementJob?
      for _ in 0..<50 {
        let update = try await first.snapshot()
        completed = update.jobs?.first { $0.id == receipt.id }
        if completed?.isActive == false { break }
        try await Task.sleep(for: .milliseconds(20))
      }
      #expect(completed?.state == "failed")
      #expect(completed?.message == "Select a catalog model.")
      #expect(!FileManager.default.fileExists(atPath: dataPath + "/servers/12345.toml"))
      await #expect(throws: ManagerError.self) { try await second.snapshot() }
      await first.close()
      let reopened = try await second.snapshot()
      #expect(reopened.catalog.models.count == snapshot.catalog.models.count)
      #expect(!FileManager.default.fileExists(atPath: dataPath + "/tls"))
      #expect(!FileManager.default.fileExists(atPath: dataPath + "/managed.toml"))
    } catch {
      await first.close()
      await second.close()
      throw error
    }
    await second.close()
  }
}
