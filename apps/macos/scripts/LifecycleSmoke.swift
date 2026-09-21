// Compile alongside Sources/PaddockClient/*.swift, not into the shipped app.
// Requires an isolated PADDOCK_DATA root, a preinstalled catalog checkpoint,
// and a .app/Contents/Helpers/paddock-runner matching the embedded library.
import Foundation

@main
struct LifecycleSmoke {
  static func main() async {
    do { try await run() }
    catch {
      FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
      exit(1)
    }
  }

  static func print(_ message: String) {
    FileHandle.standardOutput.write(Data((message + "\n").utf8))
  }

  static func run() async throws {
    guard CommandLine.arguments.count == 3,
      let root = ProcessInfo.processInfo.environment["PADDOCK_DATA"],
      URL(fileURLWithPath: root).lastPathComponent.hasPrefix("paddock-macos-lifecycle."),
      let port = UInt16(CommandLine.arguments[2]), port >= 1024
    else { throw Failure.message("Use an isolated paddock-macos-lifecycle.* data root, library path and test port.") }
    let library = URL(fileURLWithPath: CommandLine.arguments[1])
    var client = NativeManager(libraryURL: library)
    var ownedPID: UInt32?
    do {
      let initial = try await client.snapshot()
      guard initial.runners.isEmpty, initial.servers?.isEmpty != false,
        initial.catalog.models.first(where: { $0.id == "qwen3.8-27b" })?.artifacts.first(where: { $0.id == "mlx-4bit" })?.installed == true
      else { throw Failure.message("Test root must be empty apart from the installed Qwen3.8-27B MLX artifact.") }
      let started = try await settle(client, command: .create(CreateEndpointRequest(
        model: "qwen3.8-27b", artifact: "mlx-4bit", port: port, maxCtx: 4096, maxBatch: 4)))
      let live = try runner(started, port: port)
      ownedPID = live.pid
      try await health(port)
      let config = URL(fileURLWithPath: root).appending(path: "servers/\(port).toml")
      let mode = try FileManager.default.attributesOfItem(atPath: config.path)[.posixPermissions] as? NSNumber
      guard mode?.intValue == 0o600 else { throw Failure.message("Endpoint credentials are not owner-only.") }
      let stopped = try await settle(client, command: .stop(port: port, pid: live.pid))
      ownedPID = nil
      guard stopped.runners.isEmpty, stopped.servers?.first?.localOnly == true else {
        throw Failure.message("Stop did not preserve a local-only saved endpoint.")
      }
      await client.close()
      client = NativeManager(libraryURL: library)
      let reopened = try await client.snapshot()
      guard let saved = reopened.servers?.first(where: { $0.port == port }), let revision = saved.revision,
        reopened.runners.isEmpty else { throw Failure.message("Saved endpoint did not survive core reopen.") }
      let restarted = try await settle(client, command: .start(port: port, revision: revision))
      ownedPID = try runner(restarted, port: port).pid
      await client.close()
      try await health(port)
      print("PASS: healthy endpoint survives app-core close")
      client = NativeManager(libraryURL: library)
      let adopted = try await client.snapshot()
      let adoptedRunner = try runner(adopted, port: port)
      guard adoptedRunner.pid == ownedPID else { throw Failure.message("Reopened core did not adopt the same runner.") }
      _ = try await settle(client, command: .stop(port: port, pid: adoptedRunner.pid))
      ownedPID = nil
      await client.close()
      print("PASS: create, health, owner-only config, stop, reopen, restart, adoption, final stop")
    } catch {
      if let ownedPID { _ = try? await settle(client, command: .stop(port: port, pid: ownedPID)) }
      await client.close()
      throw error
    }
  }

  static func settle(_ client: NativeManager, command: ModelCommand) async throws -> ManagerSnapshot {
    let started = ContinuousClock.now
    let receipt = try await client.submit(command)
    var worstSnapshot: Duration = .zero
    for _ in 0..<360 {
      let before = ContinuousClock.now
      let snapshot = try await client.snapshot()
      worstSnapshot = max(worstSnapshot, before.duration(to: .now))
      if let job = snapshot.jobs?.first(where: { $0.id == receipt.id }), !job.isActive {
        guard job.state == "succeeded" else { throw Failure.message(job.message) }
        print("PASS: \(job.action) in \(started.duration(to: .now)); slowest management snapshot \(worstSnapshot)")
        return snapshot
      }
      try await Task.sleep(for: .seconds(1))
    }
    throw Failure.message("Lifecycle job did not settle within its test deadline.")
  }

  static func runner(_ snapshot: ManagerSnapshot, port: UInt16) throws -> RunnerInfo {
    guard let runner = snapshot.runners.first(where: { $0.port == port }) else {
      throw Failure.message("Healthy runner was missing from the inventory.")
    }
    return runner
  }

  static func health(_ port: UInt16) async throws {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/healthz")!)
    request.timeoutInterval = 10
    let (_, response) = try await URLSession.shared.data(for: request)
    guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw Failure.message("Endpoint health failed.") }
  }

  enum Failure: Error { case message(String) }
}
