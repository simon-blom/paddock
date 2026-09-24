import Foundation
import Testing

@testable import PaddockConversationCore

extension RuntimeTests {
  @Test func openStudioResolvesANewImageRunnerWithoutWaitingForTheUIProjection() async throws {
    let runtime = try await fixture()
    let before = await runtime.presentation()
    #expect(before["models"]?.array?.contains { $0["port"] == .number(12482) } == false)
    // The model starts after Studio has rendered. The OS action must use the
    // new fleet directly, not require a second click or a 32 ms render delay.
    RuntimeProtocol.state.withLock { $0.imagePeerCapabilities = editingCaps }
    _ = try await runtime.command("openEndpoint", ["port": .number(12482)])
    let state = await runtime.presentation()
    #expect(await runtime.currentFields()?["model"] == .string("local-peer"))
    #expect(state["composer"]?["imageMode"] == .bool(true))
    #expect(state["composer"]?["imageEditing"] == .bool(true))
    _ = try await runtime.command("send", ["text": .string("A red apple")])
    _ = try await settled(runtime)
    #expect(await runtime.currentFields()?["messages"]?.array?.last?["error"] == nil)
    let paths = RuntimeProtocol.state.withLock { $0.requests.compactMap { $0["path"]?.string } }
    #expect(paths.contains("/api/runners/12482/v1/images/generations"))
    #expect(!paths.contains { $0.hasSuffix("/responses") })
    await runtime.close()
  }
  @Test func openStudioSupportsSpeechAndChatWithoutCallingThemImageModels() async throws {
    let runtime = try await fixture(speechFeatures: true)
    for (port, id, image) in [(11540, "speech", false), (12481, "local", false)] {
      _ = try await runtime.command("openEndpoint", ["port": .number(Decimal(port))])
      #expect(await runtime.currentFields()?["model"] == .string(id))
      #expect(await runtime.presentation()["composer"]?["imageMode"] == .bool(image))
    }
    await runtime.close()
  }
  @Test func failedOpenStudioPreservesTheConversationAndAttachments() async throws {
    let runtime = try await fixture(speechFeatures: true)
    _ = try await runtime.command("send", ["text": .string("Keep this conversation")])
    _ = try await settled(runtime)
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("original"), "mime": .string("image/png"), "name": .string("Keep.png"),
        "size": .number(20),
      ])
    let before = await runtime.currentFields()
    // Unknown endpoint, aligner-only endpoint and invalid port values.
    for port: V in [.number(12499), .number(11541), .number(0), .number(65536), .number(1.5)] {
      await #expect(throws: ConversationFailure.self) {
        try await runtime.command("openEndpoint", ["port": port])
      }
      #expect(await runtime.currentFields() == before)
      #expect(await runtime.staged["original"] != nil)
    }
    await runtime.close()
  }
}
