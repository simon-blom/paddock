import AppKit
import Foundation
import PaddockTranscript
import SwiftUI
import Testing

@testable import PaddockClient
@testable import PaddockUI

/// Opt-in real Metal test. A synthetic conversation and a test-owned endpoint
/// live in a fresh data root; no product database or running app is touched.
@Suite("Native Studio real Metal integration", .serialized)
@MainActor struct StudioIntegrationTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_STUDIO_INTEGRATION"] == "1"))
  func generationSaveReopenFollowupAndCancel() async throws {
    let environment = ProcessInfo.processInfo.environment
    let root = try #require(environment["PADDOCK_DATA"])
    guard URL(fileURLWithPath: root).lastPathComponent.hasPrefix("paddock-macos-studio.") else {
      throw ManagerError.core("Real Studio tests require a fresh paddock-macos-studio.* root.")
    }
    let library = URL(fileURLWithPath: try #require(environment["PADDOCK_DESKTOP_TEST_LIBRARY"]))
    let assets = URL(fileURLWithPath: try #require(environment["PADDOCK_TRANSCRIPT_TEST_ASSETS"]))
    let portString = try #require(environment["PADDOCK_STUDIO_TEST_PORT"])
    let port = try #require(UInt16(portString))
    var client = NativeManager(libraryURL: library)
    var ownedPID: UInt32?
    let transcript = TranscriptSession(assetsRoot: assets)
    var chat = StudioChatModel(client: client, transcript: transcript)
    _ = NSApplication.shared
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1000, height: 800), styleMask: [.titled, .resizable],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    defer {
      transcript.close()
      window.close()
    }
    do {
      let initial = try await client.snapshot()
      #expect(initial.runners.isEmpty)
      #expect(initial.servers?.isEmpty != false)
      #expect(try await client.chat(.list).conversations?.isEmpty == true)
      let started = try await settle(
        client,
        .create(
          CreateEndpointRequest(
            model: "qwen3.8-27b", artifact: "mlx-4bit", port: port, maxCtx: 4096, maxBatch: 4)))
      let runner = try #require(started.runners.first { $0.port == port })
      ownedPID = runner.pid
      window.contentView = NSHostingView(
        rootView: LegacyStudioConversationView(
          chat: chat, draft: .constant(StudioDraft()), runners: [runner], canStart: false,
          onStart: {}))
      window.orderFrontRegardless()
      #expect(
        await chat.send(
          "Our project codename is Blue Finch. Reply with exactly: **Blue Finch**", runner: runner))
      try await finished(chat, label: "first turn")
      let first = try #require(chat.conversation)
      #expect(first.messages.count == 2)
      #expect(first.messages.last?.text.isEmpty == false)
      #expect(first.messages.last?.usage?.completionTokens != nil)
      #expect(chat.error == nil)
      try await until(seconds: 20) {
        if let error = transcript.error { throw ManagerError.core(error) }
        guard transcript.ready else { return false }
        return try await transcript.webView.evaluateJavaScript(
          "document.querySelectorAll('article.assistant').length === 1") as? Bool == true
      }
      let saved = try await client.chat(.load(first.id))
      #expect(saved.conversation == first)
      await chat.shutdown()
      await client.close()
      client = NativeManager(libraryURL: library)
      chat = StudioChatModel(client: client, transcript: TranscriptSession(assetsRoot: assets))
      await chat.open(first.id)
      #expect(chat.conversation == first)
      #expect(
        await chat.send(
          "What project codename did I give you? Reply with just that codename.", runner: runner))
      try await finished(chat, label: "follow-up")
      #expect(chat.conversation?.messages.count == 4)
      #expect(
        chat.conversation?.messages.last?.text.localizedCaseInsensitiveContains("Blue Finch")
          == true)
      #expect(chat.error == nil)
      #expect(
        await chat.send(
          "Write a detailed 2000-word essay about the history of mathematics.", runner: runner))
      await chat.cancel()
      try await finished(chat, label: "cancel")
      #expect(chat.conversation?.messages.last?.stopped == true)
      #expect(chat.conversation?.messages.last?.streaming == false)
      #expect(chat.history.contains { $0.id == first.id })
      _ = try await settle(client, .stop(port: port, pid: runner.pid))
      ownedPID = nil
      await chat.shutdown()
      await client.close()
    } catch {
      await chat.shutdown()
      if let ownedPID { _ = try? await settle(client, .stop(port: port, pid: ownedPID)) }
      await client.close()
      throw error
    }
  }
  private func finished(_ chat: StudioChatModel, label: String) async throws {
    let start = ContinuousClock.now
    try await until(seconds: 120) { !chat.busy }
    print(
      "Studio integration \(label): \(start.duration(to: .now)); output tokens \(chat.conversation?.messages.last?.usage?.completionTokens ?? 0)"
    )
  }
  private func until(seconds: TimeInterval, _ predicate: () async throws -> Bool) async throws {
    let deadline = Date().addingTimeInterval(seconds)
    while try await !predicate() {
      if Date() > deadline { throw ManagerError.core("Studio integration timed out") }
      try await Task.sleep(for: .milliseconds(20))
    }
  }
  private func settle(_ client: NativeManager, _ command: ModelCommand) async throws
    -> ManagerSnapshot
  {
    let receipt = try await client.submit(command)
    for _ in 0..<180 {
      let snapshot = try await client.snapshot()
      if let job = snapshot.jobs?.first(where: { $0.id == receipt.id }), !job.isActive {
        guard job.state == "succeeded" else { throw ManagerError.core(job.message) }
        return snapshot
      }
      try await Task.sleep(for: .seconds(1))
    }
    throw ManagerError.core("Test endpoint lifecycle timed out")
  }
}
