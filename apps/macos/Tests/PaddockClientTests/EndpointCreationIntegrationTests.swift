import Foundation
import Testing

@testable import PaddockClient

/// Opt-in live test. An isolated profile links installed weights and a runner;
/// the user's application profile, saved models and credentials are untouched.
@Suite("Live Start to Edit parity", .serialized, .timeLimit(.minutes(3)))
struct EndpointCreationIntegrationTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_START_TEST_LIBRARY"] != nil))
  func fullConfigurationSurvivesTheFirstLoad() async throws {
    let environment = ProcessInfo.processInfo.environment
    let library = try #require(environment["PADDOCK_START_TEST_LIBRARY"])
    let root = try #require(environment["PADDOCK_DATA"])
    guard URL(fileURLWithPath: root).lastPathComponent.hasPrefix("paddock-start-edit-check.") else {
      Issue.record("Live checks require an isolated profile")
      return
    }
    let manager = NativeManager(libraryURL: URL(fileURLWithPath: library))
    do {
      let prepared = try await manager.prepareEndpoint(model: "qwen3.8-27b", artifact: "mlx-4bit")
      let settings = try #require(prepared.settings)
      #expect(prepared.port == 0 && prepared.revision == nil && settings.maxBatch == 1)
      let request = CreateEndpointRequest(
        model: "qwen3.8-27b", artifact: "mlx-4bit",
        changes: [
          .maxCtx(4096), .maxBatch(1), .spec(settings.spec ?? "on"),
          .host("127.0.0.1"), .kvCacheDtype(settings.kvCacheDtype),
          .composition(
            .init(
              model: "qwen3.8-27b", artifact: "mlx-4bit", vision: settings.vision,
              drafter: settings.drafter)),
          .runtime(["temp": .number(0.7), "max_tokens": .integer(3072)]),
        ])
      let receipt = try await settle(manager, command: .create(request))
      #expect(receipt.state == "succeeded", "\(receipt.message)")
      if receipt.state == "succeeded", let port = receipt.port {
        let snapshot = try await manager.snapshot()
        let saved = snapshot.servers?.first { $0.port == port }
        let runner = snapshot.runners.first { $0.port == port }
        // Stop this exact reviewed PID before assertions which can throw.
        if let runner {
          let stopped = try await settle(manager, command: .stop(port: port, pid: runner.pid))
          #expect(stopped.state == "succeeded")
        }
        #expect(runner != nil)
        #expect(saved?.settings?.maxCtx == 4096 && saved?.settings?.maxBatch == 1)
        let options = saved?.settings?.runtimeOptions ?? []
        #expect(options.first { $0.id == "temp" }?.value == .number(0.7))
        #expect(options.first { $0.id == "max_tokens" }?.value == .integer(3072))
        #expect(saved?.settings?.spec == settings.spec)
        #expect(saved?.settings?.hasApiKey == true)
      }
    } catch {
      await manager.close()
      throw error
    }
    await manager.close()
  }

  private func settle(_ manager: NativeManager, command: ModelCommand) async throws -> ManagementJob
  {
    var job = try await manager.submit(command)
    let deadline = ContinuousClock.now + .seconds(120)
    while job.isActive && ContinuousClock.now < deadline {
      try await Task.sleep(for: .milliseconds(200))
      job = try await manager.submit(.poll(id: job.id))
    }
    return job
  }
}
