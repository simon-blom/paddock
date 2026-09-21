// Explicit live R2 test. Compile beside production PaddockClient sources into
// a private .app with our native runner; never point this at the user's data.
import Foundation

@main struct DownloadSmoke {
  enum Failure: Error { case message(String) }
  static func log(_ text: String) { FileHandle.standardOutput.write(Data((text + "\n").utf8)) }
  static func main() async {
    do { try await run() } catch {
      log("FAIL: \(error)")
      exit(1)
    }
  }
  static func run() async throws {
    guard CommandLine.arguments.count == 3,
      let root = ProcessInfo.processInfo.environment["PADDOCK_DATA"],
      URL(fileURLWithPath: root).lastPathComponent.hasPrefix("paddock-macos-download."),
      let port = UInt16(CommandLine.arguments[2]), port >= 1024
    else {
      throw Failure.message(
        "Use an isolated paddock-macos-download.* root, library path and test port")
    }
    let library = URL(fileURLWithPath: CommandLine.arguments[1])
    var client = NativeManager(libraryURL: library)
    var owned: RunnerInfo?
    do {
      let initial = try await client.snapshot()
      guard initial.servers?.isEmpty != false,
        !initial.runners.contains(where: { $0.port == port })
      else { throw Failure.message("Test endpoint is not empty") }
      let plan = try await client.downloads(.plan(model: "qwen3.8-27b", artifact: "mlx-4bit"))
      guard let plan = plan.plan, plan.fits else { throw Failure.message("No fitting plan") }
      log("PLAN: \(plan.fileCount) files, \(plan.total) bytes, selection \(plan.selection)")
      let admission = try await client.downloads(.pull(plan))
      guard let id = admission.job else { throw Failure.message("Missing receipt") }
      var didPause = false
      for _ in 0..<600 {
        let reply = try await client.downloads(.list)
        guard let job = reply.jobs.first(where: { $0.id == id }) else {
          throw Failure.message("Missing job")
        }
        if !job.active {
          throw Failure.message(
            "Transfer settled before interruption: \(job.title) \(job.status.message ?? "")")
        }
        if job.downloaded >= 16 * 1024 * 1024 {
          _ = try await client.downloads(.pause(id: id))
          didPause = true
          break
        }
        try await Task.sleep(for: .seconds(1))
      }
      guard didPause else {
        throw Failure.message("No range progress before interruption deadline")
      }
      await client.close()
      client = NativeManager(libraryURL: library)
      let restored = try await client.downloads(.list)
      guard let paused = restored.jobs.first(where: { $0.id == id }), paused.resumable else {
        throw Failure.message("Interrupted job was not restored")
      }
      log(
        "PASS: core close/reopen retained \(paused.downloaded) completed bytes and exact selection")
      let resumed = try await client.downloads(.resume(id: id))
      guard let resumedID = resumed.job else { throw Failure.message("Missing resume receipt") }
      var finished = false
      for tick in 0..<3600 {
        let reply = try await client.downloads(.list)
        guard let job = reply.jobs.first(where: { $0.id == resumedID }) else {
          throw Failure.message("Missing resumed job")
        }
        if tick % 15 == 0 { log("PROGRESS: \(job.downloaded)/\(job.total) \(job.title)") }
        if !job.active {
          guard job.complete else { throw Failure.message(job.status.message ?? job.title) }
          finished = true
          break
        }
        try await Task.sleep(for: .seconds(1))
      }
      guard finished else { throw Failure.message("Download deadline exceeded") }
      log("PASS: all R2 shards, metadata and notices verified")
      let snapshot = try await settle(
        client,
        .create(
          CreateEndpointRequest(
            model: plan.model, artifact: plan.artifact, port: port, maxCtx: 4096, maxBatch: 4)))
      guard let runner = snapshot.runners.first(where: { $0.port == port && $0.status == "ok" })
      else { throw Failure.message("Runner not healthy") }
      owned = runner
      log("PASS: downloaded MLX model started on test port \(port)")
      let receipt = try await client.chat(
        .send(conversationId: nil, runner: runner, text: "Reply with exactly: OK"))
      guard let streamID = receipt.streamId else { throw Failure.message("No response receipt") }
      var done = false
      for _ in 0..<1200 {
        let reply = try await client.chat(.poll(streamID))
        if let completion = reply.done {
          guard completion.error == nil && completion.saveError == nil else {
            throw Failure.message("Response failed")
          }
          let stored = try await client.chat(.load(completion.conversationId))
          guard
            stored.conversation?.messages.contains(where: {
              $0.role == "assistant" && !$0.text.isEmpty
            }) == true
          else { throw Failure.message("No durable answer") }
          done = true
          log("PASS: authenticated Responses generation completed and answer persisted")
          break
        }
        try await Task.sleep(for: .milliseconds(100))
      }
      guard done else { throw Failure.message("Generation deadline exceeded") }
      _ = try await settle(client, .stop(port: port, pid: runner.pid))
      owned = nil
      await client.close()
      log("PASS: download, interruption, core restart, resume, verify, run, chat, stop")
    } catch {
      if let runner = owned { _ = try? await settle(client, .stop(port: port, pid: runner.pid)) }
      await client.close()
      throw error
    }
  }
  static func settle(_ client: NativeManager, _ command: ModelCommand) async throws
    -> ManagerSnapshot
  {
    let receipt = try await client.submit(command)
    for _ in 0..<360 {
      let snapshot = try await client.snapshot()
      if let job = snapshot.jobs?.first(where: { $0.id == receipt.id }), !job.isActive {
        guard job.state == "succeeded" else { throw Failure.message(job.message) }
        return snapshot
      }
      try await Task.sleep(for: .seconds(1))
    }
    throw Failure.message("Lifecycle deadline exceeded")
  }
}
