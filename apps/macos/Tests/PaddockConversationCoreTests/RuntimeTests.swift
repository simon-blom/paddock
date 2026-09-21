import Foundation
import PaddockClient
import Synchronization
import Testing

@testable import PaddockConversationCore

@Suite("Native Studio cutover", .serialized)
struct RuntimeTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  func fixture(speechFeatures: Bool = false, history: [O] = []) async throws -> NativeStudioRuntime
  {
    RuntimeProtocol.state.withLock {
      $0 = .init()
      $0.speechFeatures = speechFeatures
      $0.documents = Dictionary(uniqueKeysWithValues: history.map { ($0["id"]!.string!, $0) })
    }
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43210", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [RuntimeProtocol.self]
    let transport = try NativeConversationTransport(host: host, configuration: configuration)
    let runtime = NativeStudioRuntime(transport: transport) { _ in }
    try await runtime.start()
    return runtime
  }
  func settled(_ runtime: NativeStudioRuntime) async throws -> O {
    for _ in 0..<500 {
      let state = await runtime.presentation()
      if state["busy"]?.bool == false { return state }
      try await Task.sleep(for: .milliseconds(10))
    }
    throw ConversationFailure.invalid("Native runtime did not settle")
  }
  @Test func cloudHTTPFailureKeepsRecoveryMetadataAndIsNotAutomaticallyRetried() async throws {
    let runtime = try await fixture()
    RuntimeProtocol.state.withLock { $0.responseFailure = true }
    _ = try await runtime.command("models", ["ids": .array([.string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("Provider failure fixture")])
    let state = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    let raw = try #require(messages.last?["error"]?.string)
    let error = try JSONDecoder().decode(V.self, from: Data(raw.utf8))
    #expect(error["code"]?.integer == 429)
    #expect(error["metadata"]?["provider_name"]?.string == "DeepInfra")
    #expect(error["metadata"]?["action"]?.string == "openrouter_integrations")
    #expect(messages.last?["streaming"]?.bool != true)
    #expect(
      RuntimeProtocol.state.withLock {
        $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }.count
      } == 1)
    let id = try #require(state["conversation"]?["id"]?.string)
    #expect(
      RuntimeProtocol.state.withLock {
        $0.documents[id]?["messages"]?.array?.last?["error"]?.string
      } == raw)
    await runtime.close()
  }
  @Test func artifactPanelStateAndAvailableFilesSurviveRevisit() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Make a page")])
    let state = try await settled(runtime)
    let id = try #require(state["conversation"]?["id"]?.string)
    let artifact: V = .object([
      "id": .string("art_012345abcdef"), "kind": .string("html"), "title": .string("Page"),
      "model": .string("local"), "versions": .number(1), "updatedAt": .number(1),
    ])
    RuntimeProtocol.state.withLock { $0.artifacts = [artifact] }
    _ = try await runtime.command("refresh")
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.count == 1)
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == true)
    _ = try await runtime.command(
      "artifactsPane", ["open": .bool(false), "conversationId": .string(id)])
    #expect(
      RuntimeProtocol.state.withLock { $0.documents[id]?["artifactsPaneOpen"]?.bool } == false)
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": .string(id)])
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.count == 1)
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == false)
    _ = try await runtime.command("graphPanel", ["open": .bool(true)])
    _ = try await runtime.command(
      "artifactsPane", ["open": .bool(true), "conversationId": .string(id)])
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == true)
    #expect(await runtime.graphVisible == false)
    await #expect(throws: ConversationFailure.self) {
      _ = try await runtime.command(
        "artifactsPane", ["open": .bool(false), "conversationId": .string("stale-chat")])
    }
    RuntimeProtocol.state.withLock { $0.failSave = true }
    await #expect(throws: (any Error).self) {
      _ = try await runtime.command(
        "artifactsPane", ["open": .bool(false), "conversationId": .string(id)])
    }
    #expect(await runtime.presentation()["nativeArtifactsPaneOpen"]?.bool == true)
    await runtime.close()
  }
  @Test func completedArtifactRefreshIsCoalescedAndCannotLeakIntoAnotherChat() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Make a page")])
    _ = try await settled(runtime)
    RuntimeProtocol.state.withLock {
      $0.artifacts = [.object(["id": .string("art_012345abcdef"), "kind": .string("html")])]
    }
    let before = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/artifacts") == true }.count
    }
    for _ in 0..<8 { await runtime.scheduleArtifactRefresh() }
    try await Task.sleep(for: .milliseconds(200))
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.count == 1)
    #expect(
      RuntimeProtocol.state.withLock {
        $0.requests.filter { $0["path"]?.string?.hasSuffix("/artifacts") == true }.count
      } == before + 1)
    await runtime.scheduleArtifactRefresh()
    _ = try await runtime.command("newChat")
    try await Task.sleep(for: .milliseconds(200))
    #expect(await runtime.presentation()["nativeArtifacts"]?.array?.isEmpty == true)
    await runtime.close()
  }
  @Test func newChatSurfacesAboveLoadedUnpinnedHistoryWithoutRestartOrRefresh() async throws {
    var history: [O] = (0..<32).map { i in
      [
        "id": .string("old-\(i)"), "title": .string("Old \(i)"), "model": .string("local"),
        "updatedAt": .number(Decimal(i)), "pinned": .bool(false),
      ]
    }
    history.append([
      "id": .string("pinned"), "title": .string("Pinned"), "updatedAt": .number(0),
      "pinned": .bool(true),
    ])
    let runtime = try await fixture(history: history)
    _ = try await runtime.command("send", ["text": .string("Brand new chat")])
    let state = try await settled(runtime)
    let id = try #require(state["conversation"]?["id"]?.string)
    for rows in [state["history"]?.array, state["library"]?["rows"]?.array] {
      #expect(rows?.prefix(2).compactMap { $0["id"]?.string } == ["pinned", id])
    }
    // A follow-up save, title edit and explicit reload must retain that order.
    _ = try await runtime.command(
      "renameChat", ["id": .string(id), "title": .string("Renamed new chat")])
    _ = try await runtime.command("refresh")
    let refreshed = await runtime.presentation()
    #expect(
      refreshed["history"]?.array?.prefix(2).compactMap { $0["id"]?.string } == ["pinned", id])
    await runtime.close()
  }
  @Test func liveProjectionShowsNextUtteranceBeforeItsFinalMetadata() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("models", ["ids": .array([.string("speech")])])
    let mapping = try await runtime.beginLive()
    let mid = try #require(mapping["speech"])
    let meta: O = [
      "words": .array([
        .object([
          "word": .string("Hej"), "start": .number(0), "end": .number(0.5),
          "confidence": .number(0.3),
        ])
      ])
    ]
    try await runtime.liveUpdate(messageID: mid, text: "Hej nästa mening", transcript: meta)
    let state = await runtime.presentation()
    let words = try #require(
      state["nativeTranscript"]?["messages"]?.array?.last?["speech"]?["words"]?.array)
    #expect(words.compactMap { $0["word"]?.string } == ["Hej", "nästa", "mening"])
    #expect(words.allSatisfy { $0["start"] == nil && $0["confidence"] == nil })
    let doc = try #require(await runtime.currentFields())
    #expect(
      doc["messages"]?.array?.last?["transcript"]?.object == meta,
      "Presentation must not erase completed utterance metadata")
    await runtime.close()
  }
  @Test func compareUsesExactLocalAndCloudRoutesAndDurableTerminalTails() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    let id = UUID().uuidString
    let accepted = try await runtime.command("send", ["text": .string("Hello")], id: id)
    #expect(accepted["accepted"]?.bool == true)
    _ = try await runtime.command("send", ["text": .string("MUST NOT DUPLICATE")], id: id)
    let state = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    #expect(messages.count == 3)
    #expect(messages[1]["group"] == messages[2]["group"])
    #expect(state["nativeTranscript"]?["messages"]?.array?.count == 3)
    #expect(messages[1]["content"]?.array?.first?["text"]?.string == "Local final tail 🦊")
    #expect(messages[2]["content"]?.array?.first?["text"]?.string == "Cloud final tail 🦊")
    let requests = RuntimeProtocol.state.withLock { $0.requests }
    #expect(
      requests.contains {
        $0["path"]?.string == "/api/runners/12481/v1/responses"
          && $0["body"]?["model"]?.string == "local"
      })
    #expect(
      requests.contains {
        $0["path"]?.string == "/api/cloud/ep/v1/responses"
          && $0["body"]?["model"]?.string == "remote"
      })
    let saved = RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!] }
    #expect(saved?["messages"] == doc["messages"])
    await runtime.close()
  }
  @Test func runDetailsUseTheActualRequestSnapshotAndSurviveConfigurationChanges() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "settings", ["params": .object(["temperature": .number(0.4), "thinking": .bool(false)])])
    _ = try await runtime.command("send", ["text": .string("Run details")])
    _ = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let assistant = try #require(doc["messages"]?.array?.last)
    let run = try #require(assistant["run"]?.object)
    let request = try #require(
      RuntimeProtocol.state.withLock {
        $0.requests.first { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
      })
    #expect(run["params"]?["maxTokens"] == (request["max_output_tokens"] ?? .null))
    #expect(
      (request["max_output_tokens"]?.integer ?? 0) > 1024,
      "Model maximum must not fall through to the runner's API default")
    #expect(
      request["max_output_tokens"]?.integer == 4096,
      "Local runner clamps exact remaining context; no extra client 1024 reserve")
    #expect(run["params"]?["temperature"]?.double == 0.4)
    #expect(run["tools"]?.array == [.string("artifacts")])
    _ = try await runtime.command("settings", ["params": .object(["temperature": .number(0.9)])])
    let state = await runtime.presentation()
    let chrome = try #require(state["nativeTranscript"]?["messages"]?.array?.last?["chrome"])
    #expect(chrome["sections"]?.array?.count == 2)
    #expect(chrome["footer"]?.string?.contains("5 tokens") == true)
    #expect(
      chrome["sections"]?.array?.first?["rows"]?.array?.contains {
        $0["label"]?.string == "Sampling" && $0["value"]?.string?.contains("temp 0.4") == true
      } == true)
    #expect(await runtime.currentFields()?["messages"]?.array?.last?["run"]?.object == run)
    await runtime.close()
  }

  @Test func explicitReplyLimitSurvivesAndCompareUsesEachProviderCeiling() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "preferencesSave",
      ["changes": .object(["maxTokens": .number(1536)]), "expected": .object(["maxTokens": .null])])
    _ = try await runtime.command("send", ["text": .string("Explicit limit")])
    _ = try await settled(runtime)
    let explicit = RuntimeProtocol.state.withLock {
      $0.requests.first { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
    }
    #expect(explicit?["max_output_tokens"]?.integer == 1536)
    _ = try await runtime.command("newChat")
    _ = try await runtime.command(
      "preferencesSave",
      ["changes": .object(["maxTokens": .null]), "expected": .object(["maxTokens": .number(1536)])])
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("Compare limits")])
    _ = try await settled(runtime)
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }.suffix(2)
    }
    let cloud = requests.first { $0["path"]?.string?.contains("/cloud/") == true }?["body"]
    let local = requests.first { $0["path"]?.string?.contains("/runners/") == true }?["body"]
    #expect(cloud?["max_output_tokens"]?.integer == 2048)
    #expect((local?["max_output_tokens"]?.integer ?? 0) > 2048)
    await runtime.close()
  }

  @Test func compareMakerProviderAndBadgesSurviveFleetRemovalWithoutChangingTheRoute() async throws
  {
    let runtime = try await fixture()
    let kimi = "cloud:ep:moonshotai/kimi-k3@deepinfra/bf16"
    let muse = "cloud:ep:meta/muse"
    _ = try await runtime.command("models", ["ids": .array([.string(kimi), .string(muse)])])
    _ = try await runtime.command("send", ["text": .string("Compare identity")])
    let state = try await settled(runtime)
    let lanes = try #require(state["nativeTranscript"]?["messages"]?.array).dropFirst()
    #expect(
      lanes.map { $0["chrome"]?["modelName"]?.string } == ["Kimi K3 (deepinfra/bf16)", "Muse"])
    #expect(lanes.map { $0["chrome"]?["vendor"]?.string } == ["Moonshot", "Meta"])
    #expect(lanes.allSatisfy { $0["chrome"]?["tools"]?.array == [.string("artifacts")] })
    #expect(lanes.allSatisfy { $0["chrome"]?["fastest"]?.bool == false })
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
    }
    #expect(requests.count == 2)
    #expect(
      Set(requests.compactMap { $0["body"]?["model"]?.string }) == [
        "moonshotai/kimi-k3@deepinfra/bf16", "meta/muse",
      ])
    #expect(requests.allSatisfy { $0["path"]?.string == "/api/cloud/ep/v1/responses" })
    let saved = try #require(
      await runtime.currentFields()?["messages"]?.array?.dropFirst().first?.object)
    #expect(saved["run"]?["modelName"]?.string == "Kimi K3 (deepinfra/bf16)")
    #expect(saved["run"]?["vendor"]?.string == "Moonshot")

    RuntimeProtocol.state.withLock { $0.cloudEnabled = false }
    _ = try await runtime.command("refresh")
    let restored = await runtime.messageProjection(saved, controls: nil)
    #expect(restored["chrome"]?["modelName"] == lanes.first?["chrome"]?["modelName"])
    #expect(restored["chrome"]?["vendor"] == lanes.first?["chrome"]?["vendor"])
    var legacy = saved
    legacy["run"] = .object(["contended": .bool(true), "tools": .array([.string("filesystem")])])
    legacy["toolCalls"] = .array([])
    let projected = await runtime.messageProjection(legacy, controls: nil)
    #expect(projected["chrome"]?["modelName"]?.string == "kimi k3")
    #expect(projected["chrome"]?["vendor"]?.string == "Moonshot")
    #expect(projected["contended"]?.bool == true)
    #expect(
      projected["chrome"]?["tools"]?.array == [.string("filesystem")],
      "Tools badge describes the run, not whether a tool was called")
    await runtime.close()
  }

  @Test func discoveryDoesNotAppearAsAnArtifactInvocationInNewOrSavedChats() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Tool discovery")])
    _ = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let mid = try #require(doc["leafId"]?.string)
    let discovery: O = [
      "id": .string("discovery"), "type": .string("mcp_list_tools"),
      "server_label": .string("artifacts"),
    ]
    try await runtime.applyOutputItem(discovery, messageID: mid, done: true)
    #expect(await runtime.currentFields()?["messages"]?.array?.last?["toolCalls"] == nil)
    // Previous builds saved discovery. Rendering must filter it without a DB rewrite.
    try await runtime.updateMessage(mid) { $0["toolCalls"] = .array([.object(discovery)]) }
    let search: O = [
      "id": .string("search"), "type": .string("mcp_call"), "name": .string("mcp_search_tools"),
      "status": .string("completed"),
    ]
    try await runtime.applyOutputItem(search, messageID: mid, done: true)
    let state = await runtime.presentation()
    #expect(
      state["nativeTranscript"]?["messages"]?.array?.last?["toolCalls"]?.array?.map {
        $0["name"]?.string
      } == ["mcp_search_tools"])
    #expect(
      await runtime.currentFields()?["messages"]?.array?.last?["toolCalls"]?.array?.count == 2)
    await runtime.close()
  }

  @Test func failedAdmissionKeepsDraftAndDoesNotCallAModel() async throws {
    let runtime = try await fixture()
    let before = await runtime.currentFields()
    RuntimeProtocol.state.withLock { $0.failSave = true }
    await #expect(throws: (any Error).self) {
      try await runtime.command("send", ["text": .string("Keep me")])
    }
    #expect(await runtime.currentFields() == before)
    #expect(
      !RuntimeProtocol.state.withLock {
        $0.requests.contains { $0["path"]?.string?.hasSuffix("/responses") == true }
      })
    await runtime.close()
  }
  @Test func retryEditBranchesAndUnknownFieldsSurvive() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("send", ["text": .string("Original question")])
    _ = try await settled(runtime)
    var doc = try #require(await runtime.currentFields())
    doc["future"] = .object(["exact": .number(9_007_199_254_740_993)])
    RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!] = doc }
    _ = try await runtime.command("open", ["id": doc["id"]!])
    let oldLeaf = doc["leafId"]!
    _ = try await runtime.command(
      "messageAction",
      [
        "action": .string("retry"), "conversationId": doc["id"]!, "leafId": oldLeaf,
        "messageId": oldLeaf,
      ])
    _ = try await settled(runtime)
    var next = try #require(await runtime.currentFields())
    #expect(next["messages"]?.array?.count == 3)
    #expect(next["future"] == doc["future"])
    let user = doc["messages"]!.array![0]["id"]!
    _ = try await runtime.command(
      "messageAction",
      [
        "action": .string("edit"), "conversationId": doc["id"]!, "leafId": next["leafId"]!,
        "messageId": user, "originalText": .string("Original question"),
        "text": .string("Edited question"),
      ])
    _ = try await settled(runtime)
    next = try #require(await runtime.currentFields())
    #expect(next["messages"]?.array?.count == 5)
    let active = try ConversationDocument(fields: next).activeMessages
    #expect(ConversationDocument.text(active[0]) == "Edited question")
    #expect(active.count == 2)
    await #expect(throws: ConversationFailure.stale) {
      try await runtime.command(
        "messageAction",
        [
          "action": .string("retry"), "conversationId": doc["id"]!, "leafId": oldLeaf,
          "messageId": oldLeaf,
        ])
    }
    await runtime.close()
  }
  @Test func compareContextNeverReplaysTheOtherLanesAnswer() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "models", ["ids": .array([.string("local"), .string("cloud:ep:remote")])])
    _ = try await runtime.command("send", ["text": .string("First")])
    _ = try await settled(runtime)
    _ = try await runtime.command("send", ["text": .string("Second")])
    _ = try await settled(runtime)
    let requests = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["path"]?.string?.hasSuffix("/responses") == true }
    }
    #expect(requests.count == 4)
    for request in requests.suffix(2) {
      let body = String(decoding: try JSONEncoder().encode(request["body"]!), as: UTF8.self)
      if request["path"]?.string?.contains("/cloud/") == true {
        #expect(!body.contains("Local final"))
      } else {
        #expect(!body.contains("Cloud final"))
      }
    }
    await runtime.close()
  }
  @Test func historyTitleAndSamplingAreNativeAndAcknowledged() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command(
      "settings", ["params": .object(["temperature": .number(0.4), "thinking": .bool(false)])])
    _ = try await runtime.command("send", ["text": .string("A question")])
    _ = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    _ = try await runtime.command(
      "renameChat", ["id": doc["id"]!, "title": .string("My full title")])
    _ = try await runtime.command("pinChat", ["id": doc["id"]!])
    let state = await runtime.presentation()
    #expect(state["history"]?.array?.first?["title"]?.string == "My full title")
    #expect(state["history"]?.array?.first?["pinned"]?.bool == true)
    let body = RuntimeProtocol.state.withLock {
      $0.requests.first { $0["path"]?.string?.hasSuffix("/responses") == true }?["body"]
    }
    #expect(body?["temperature"]?.double == 0.4)
    #expect(body?["chat_template_kwargs"]?["enable_thinking"]?.bool == false)
    await runtime.close()
  }
  @Test func segmentOnlySpeechFinishesWithFinalWordAndOriginalRecording() async throws {
    let runtime = try await fixture()
    _ = try await runtime.command("models", ["ids": .array([.string("speech")])])
    let targets = try await runtime.audioTargets(dictation: false)
    #expect(targets.first?["transcription"]?["paddock_verbose"] == nil)
    let mapping = try await runtime.beginLive()
    let messageID = try #require(mapping["speech"])
    try await runtime.liveUpdate(
      messageID: messageID, text: "Testar funktionen med", transcript: [:])
    try await runtime.liveUpdate(
      messageID: messageID, text: "Testar funktionen med ljudinspelning.", transcript: [:])
    let clip: O = [
      "type": .string("audio"), "attachmentId": .string("original-wav"),
      "mime": .string("audio/wav"), "durationS": .number(4.0419375),
    ]
    try await runtime.finishLive(messages: [messageID], clip: clip)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    #expect(messages[0]["content"]?.array == [.object(clip)])
    #expect(
      messages[1]["content"]?.array?.first?["text"]?.string
        == "Testar funktionen med ljudinspelning.")
    #expect(messages[1]["streaming"]?.bool == false)
    #expect(messages[1]["error"] == nil)
    #expect(
      RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!]?["messages"] }
        == doc["messages"])
    await runtime.close()
  }
  @Test func speechTaskInstructionAlignmentAndRealtimeMetricsAreWiredEndToEnd() async throws {
    let runtime = try await fixture(speechFeatures: true)
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("fixture-audio"), "name": .string("Recording.wav"),
        "mime": .string("audio/wav"), "size": .number(44), "durationS": .number(2),
      ])
    _ = try await runtime.command(
      "send",
      [
        "text": .string("Identify speakers"),
        "attachments": .array([.object(["id": .string("fixture-audio")])]),
      ])
    let state = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let assistant = try #require(doc["messages"]?.array?.last)
    #expect(assistant["content"]?.array?.first?["text"]?.string == "Hello world")
    #expect(assistant["transcript"]?["words"]?.array?.last?["end"] == .number(1.5))
    #expect(assistant["transcript"]?["words"]?.array?.first?["confidence"] == .number(0.9))
    #expect(assistant["transcript"]?["wordsFrom"]?.string == "aligner")
    #expect((assistant["usage"]?["ms"]?.double ?? 0) > 0)
    #expect(
      state["nativeTranscript"]?["messages"]?.array?.last?["chrome"]?["footer"]?.string?.contains(
        "× realtime") == true)
    let requests = RuntimeProtocol.state.withLock { $0.requests }
    let form = try #require(
      requests.first { $0["path"]?.string?.hasSuffix("/transcriptions") == true }?["multipart"]?
        .string)
    #expect(form.contains("name=\"prompt\"\r\n\r\nIdentify speakers"))
    #expect(!form.contains("timestamp_granularities[]"))
    #expect(requests.contains { $0["path"]?.string == "/api/runners/11541/v1/audio/alignments" })
    #expect(!requests.contains { $0["path"]?.string?.hasSuffix("/responses") == true })
    await runtime.close()
  }
  @Test func failedLiveLaneDoesNotPoisonSuccessfulTranscriptOrOriginal() async throws {
    let runtime = try await fixture(speechFeatures: true)
    _ = try await runtime.command("models", ["ids": .array([.string("local"), .string("speech")])])
    let lanes = try await runtime.beginLive()
    let successful = try #require(lanes["local"])
    let failed = try #require(lanes["speech"])
    try await runtime.liveUpdate(
      messageID: successful, text: "Hello world",
      transcript: ["language": .string("en"), "durationS": .number(2)])
    try await runtime.liveFailure(messageID: failed, error: "Model disconnected")
    let clip: O = [
      "type": .string("audio"), "attachmentId": .string("fixture-audio"),
      "mime": .string("audio/wav"), "durationS": .number(2),
    ]
    try await runtime.finishLive(messages: [successful, failed], clip: clip)
    let doc = try #require(await runtime.currentFields())
    let messages = try #require(doc["messages"]?.array)
    let good = try #require(messages.first { $0["id"]?.string == successful })
    let bad = try #require(messages.first { $0["id"]?.string == failed })
    #expect(good["error"] == nil)
    #expect(good["streaming"]?.bool == false && bad["streaming"]?.bool == false)
    #expect(good["content"]?.array?.first?["text"]?.string == "Hello world")
    #expect(good["transcript"]?["wordsFrom"]?.string == "aligner")
    #expect(bad["error"]?.string == "Model disconnected")
    #expect(messages.first?["content"]?.array?.first?["attachmentId"]?.string == "fixture-audio")
    #expect(
      RuntimeProtocol.state.withLock { $0.documents[doc["id"]!.string!]?["messages"] }
        == doc["messages"])
    await runtime.close()
  }
}

private final class RuntimeProtocol: URLProtocol, @unchecked Sendable {
  typealias V = ConversationValue
  typealias O = [String: V]
  struct State: Sendable {
    var documents: [String: O] = [:]
    var requests: [O] = []
    var failSave = false
    var speechFeatures = false
    var cloudEnabled = true
    var artifacts: [V] = []
    var responseFailure = false
  }
  static let state = Mutex(State())
  override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "127.0.0.1" }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let path = request.url!.path
    let method = request.httpMethod ?? "GET"
    var bytes = request.httpBody ?? Data()
    if bytes.isEmpty, let stream = request.httpBodyStream {
      stream.open()
      defer { stream.close() }
      var buffer = [UInt8](repeating: 0, count: 4096)
      while stream.hasBytesAvailable {
        let n = stream.read(&buffer, maxLength: buffer.count)
        if n <= 0 { break }
        bytes.append(contentsOf: buffer.prefix(n))
      }
    }
    let body = (try? JSONDecoder().decode(V.self, from: bytes)) ?? .null
    let response: (Int, String, Data) = Self.state.withLock { state in
      state.requests.append([
        "path": .string(path), "body": body,
        "multipart": .string(String(decoding: bytes, as: UTF8.self)),
      ])
      var value: V = .object([:])
      var status = 200
      if state.speechFeatures, path.hasPrefix("/api/attachments/") {
        return (200, "audio/wav", Data(repeating: 0, count: 44))
      } else if state.speechFeatures, path == "/api/runners" {
        value = .array([
          .object([
            "model": .string("local"), "display": .string("Granite Speech Plus"),
            "port": .number(12481), "status": .string("ok"),
          ]),
          .object([
            "asr": .string("speech"), "display": .string("Whisper"), "port": .number(11540),
            "status": .string("ok"),
          ]),
          .object(["aligner": .string("aligner"), "port": .number(11541), "status": .string("ok")]),
        ])
      } else if state.speechFeatures, path == "/api/runners/12481/server" {
        value = .object([
          "audio": .bool(true), "timestamp_granularities": .array([.string("word")]),
          "include": .array([.string("logprobs")]), "transcription_max_clip_s": .number(120),
          "realtime_transcription": .object(["supported": .bool(true), "enrichment": .bool(false)]),
        ])
      } else if state.speechFeatures, path == "/api/runners/11541/server" {
        value = .object(["aligner": .string("aligner"), "alignment_max_clip_s": .number(120)])
      } else if state.speechFeatures, path.hasSuffix("/transcriptions") {
        let terminal =
          "{\"type\":\"transcript.text.done\",\"text\":\"Hello world\",\"paddock_verbose\":{\"duration\":2,\"words\":[{\"word\":\"Hello\",\"paddock_confidence\":0.9},{\"word\":\"world\"}]}}"
        return (
          200, "text/event-stream",
          Data(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"Hello world\"}\n\ndata: \(terminal)\n\n"
              .utf8)
        )
      } else if state.speechFeatures, path.hasSuffix("/alignments") {
        return (
          200, "application/json",
          Data(
            "{\"words\":[{\"word\":\"Hello\",\"start\":0.2,\"end\":0.8},{\"word\":\"world\",\"start\":0.8,\"end\":1.5}],\"language_supported\":true}"
              .utf8)
        )
      } else if path == "/api/settings" {
        value = .object(["macos_studio_preferences": .object(["pk_auto_title": .string("off")])])
      } else if path == "/api/runners" {
        value = .array([
          .object([
            "model": .string("local"), "display": .string("Local model"), "port": .number(12481),
            "status": .string("ok"),
          ]),
          .object([
            "asr": .string("speech"), "display": .string("Whisper"), "port": .number(11540),
            "status": .string("ok"),
          ]),
        ])
      } else if path == "/api/cloud" {
        value = .array(
          state.cloudEnabled
            ? [
              .object([
                "id": .string("ep"), "name": .string("Fixture cloud"), "hasKey": .bool(true),
                "models": .array([
                  .object([
                    "id": .string("remote"), "display": .string("Remote model"),
                    "vision": .bool(false), "ctx": .number(131072), "maxOut": .number(2048),
                  ]),
                  .object([
                    "id": .string("moonshotai/kimi-k3"), "display": .string("MoonshotAI: Kimi K3"),
                    "provider": .string("deepinfra/bf16"),
                  ]),
                  .object(["id": .string("meta/muse"), "display": .string("Meta: Muse")]),
                ]),
              ])
            ] : [])
      } else if path == "/api/runners/11540/server" {
        value = .object([
          "asr": .string("speech"), "timestamp_granularities": .array([.string("segment")]),
        ])
      } else if path.hasSuffix("/server") {
        value = .object([
          "max_ctx": .number(4096), "default_max_output_tokens": .number(1024),
          "reasoning": .string("toggle"), "vision": .bool(true),
        ])
      } else if path == "/api/conversations" {
        value = .array(state.documents.values.map(V.object))
      } else if path.hasSuffix("/artifacts") {
        value = .array(state.artifacts)
      } else if path.hasPrefix("/api/conversations/") {
        let id = String(path.split(separator: "/").last!)
        if method == "PUT" {
          if state.failSave { status = 503 } else { state.documents[id] = body.object }
        } else if let doc = state.documents[id] {
          value = .object(doc)
        } else {
          status = 404
        }
      } else if path.hasSuffix("/responses") {
        if state.responseFailure {
          let error =
            #"{"error":{"message":"DeepInfra: model is temporarily rate-limited upstream.","metadata":{"provider_name":"DeepInfra","provider_error_code":"engine_overloaded","action":"openrouter_integrations"}}}"#
          return (429, "application/json", Data(error.utf8))
        }
        let answer = path.contains("/cloud/") ? "Cloud final tail 🦊" : "Local final tail 🦊"
        let terminal: O = [
          "type": .string("response.completed"),
          "response": .object([
            "status": .string("completed"),
            "output": .array([
              .object([
                "type": .string("message"),
                "content": .array([
                  .object(["type": .string("output_text"), "text": .string(answer)])
                ]),
              ])
            ]), "usage": .object(["input_tokens": .number(10), "output_tokens": .number(5)]),
          ]),
        ]
        let end = String(decoding: try! JSONEncoder().encode(terminal), as: UTF8.self)
        return (
          200, "text/event-stream",
          Data(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\ndata: \(end)\n\n"
              .utf8)
        )
      }
      return (status, "application/json", try! JSONEncoder().encode(value))
    }
    client?.urlProtocol(
      self,
      didReceive: HTTPURLResponse(
        url: request.url!, statusCode: response.0, httpVersion: "HTTP/1.1",
        headerFields: ["Content-Type": response.1])!, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: response.2)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {}
}
