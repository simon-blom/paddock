import AppKit
import Foundation
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Speech setup selection", .serialized) @MainActor
struct SpeechSetupTests {
  @Test func startUsesAutomaticAddressAndBackendLimitsForAdvancedDefaults() throws {
    let snapshot = try speechSetupFixture()
    let view = StartModelView(
      snapshot: snapshot, model: snapshot.catalog.models.first?.id, purpose: .speech
    ) { _ in false }
    #expect(view.validation == nil && !view.initialArtifact.isEmpty)
    #expect(
      view.request == nil, "Start is blocked until Rust has prepared the full settings schema")
    let runtime = try ManagerWire.decode(
      ArtifactRuntime.self,
      from: Data(
        #"{"memory":{"max_ctx":448,"max_batch":16}}"#.utf8))
    let defaults = StartModelView.defaults(runtime)
    #expect(defaults.context == 448 && defaults.batch == 1)
    let chat = StartModelView(snapshot: try endpointFixture()) { _ in false }
    #expect(chat.request == nil)
    #expect(
      StartModelView.defaults(nil).batch == 1,
      "A native personal launch must not inherit the runner's 32-slot default")
    #expect(StartModelView.defaults(nil).context == 32768)
  }

  @Test func installedChatModelIsNotASpeechFallback() throws {
    let snapshot = try endpointFixture()
    let qwen = try #require(snapshot.catalog.models.first)
    #expect(!StartModelView.artifacts(qwen, snapshot: snapshot).isEmpty)
    #expect(StartModelView.artifacts(qwen, snapshot: snapshot, purpose: .speech).isEmpty)
    for explicit in [nil, qwen.id] {
      let view = StartModelView(
        snapshot: snapshot, model: explicit, purpose: .speech
      ) { _ in
        Issue.record("Setup must not start a model on open")
        return false
      }
      #expect(
        view.validation == "Select downloaded speech-to-text weights compatible with this Mac.")
      #expect(view.speechUnavailable)
    }
  }

  @Test func exactExportCapabilityAndBackendWinOverFamilyAndName() throws {
    let snapshot = try speechSetupFixture()
    let model = try #require(snapshot.catalog.models.first)
    #expect(
      StartModelView.artifacts(model, snapshot: snapshot, purpose: .speech).map(\.id)
        == ["asr", "legacy", "unqualified"])
    // ASR can be generative (chat + transcription). TTS and alignment cannot
    // serve dictation, nor can an explicitly empty capability list.
    for id in ["chat", "empty", "align", "tts", "cuda", "download"] {
      let view = StartModelView(
        snapshot: snapshot, model: model.id, artifact: id, purpose: .speech
      ) { _ in false }
      #expect(view.validation != nil, "Rejected \(id)")
    }
    #expect(
      StartModelView(snapshot: snapshot, purpose: .speech) { _ in false }.initialModel.isEmpty,
      "Speech setup, like text setup, requires an explicit model choice")
  }

  @Test func catalogOffersDownloadsButNeverUnrelatedExportsOrFallbacks() throws {
    let snapshot = try speechSetupFixture()
    let entries = LibraryCatalog.entries(
      catalog: snapshot.catalog, backend: "metal", purpose: .speech)
    let entry = try #require(entries.first)
    #expect(entry.artifacts.map(\.id) == ["asr", "legacy", "unqualified", "download"])
    #expect(
      ModelStartPurpose.speech.weights(entry.model, backend: "metal").map(\.id)
        == entry.artifacts.map(\.id), "Detail options use the same restriction as the list")
    let stale = LibrarySelection(model: entry.id, artifact: "chat")
    #expect(
      LibraryCatalog.selection(stale, in: entries, backend: "metal", purpose: .speech)?.artifact
        == "asr")
    #expect(
      LibraryCatalog.entries(
        catalog: snapshot.catalog, backend: "metal", query: "chat", purpose: .speech
      ).isEmpty)
    #expect(
      LibraryCatalog.entries(
        catalog: snapshot.catalog, backend: "metal", downloadedOnly: true, purpose: .speech
      )
      .first?.artifacts.map(\.id) == ["asr", "legacy", "unqualified"])
    #expect(
      !LibraryRecommendation.explanation(model: entry.model, backend: "metal", purpose: .speech)
        .contains("catalog default is"))
  }

  @Test func legacyQualificationLabelsCannotBlockSpeechSetup() throws {
    let snapshot = try speechSetupFixture(onlyUnavailable: true)
    let entries = LibraryCatalog.entries(
      catalog: snapshot.catalog, backend: "metal", purpose: .speech)
    #expect(entries.first?.artifacts.map(\.id) == ["unqualified"])
    #expect(entries.first?.artifacts.first?.supportNotice == nil)
    let view = StartModelView(
      snapshot: snapshot, model: snapshot.catalog.models.first?.id, purpose: .speech
    ) { _ in false }
    #expect(!view.speechUnavailable && view.validation == nil)
    let available = StartModelView(snapshot: try speechSetupFixture(), purpose: .speech) { _ in
      false
    }
    #expect(!available.speechUnavailable)
  }

  @Test func browseIntentPersistsWithoutDiscardingStudioDestination() {
    var navigation = WorkspaceNavigation()
    navigation.studio = .chats
    navigation.showModelLibrary(purpose: .speech)
    #expect(navigation.mode == .manager && navigation.manager == .models)
    #expect(navigation.libraryPurpose == .speech && navigation.studio == .chats)
    navigation.showManager(.downloads)
    navigation.showManager(.models)
    #expect(navigation.libraryPurpose == .speech)
    navigation.showModelLibrary(purpose: .all)
    #expect(navigation.libraryPurpose == .all)
    #expect(DesktopAction.startSpeechModel != .startModel)
  }

  @Test func emptySpeechSetupFitsBothAppearancesWithoutOpeningAWindow() throws {
    _ = NSApplication.shared
    let snapshot = try endpointFixture()
    for dark in [false, true] {
      let host = NSHostingController(
        rootView:
          StartModelView(snapshot: snapshot, purpose: .speech, onBrowse: {}) { _ in false }
          .environment(\.colorScheme, dark ? .dark : .light))
      let size = host.sizeThatFits(in: CGSize(width: 680, height: 740))
      #expect(
        size.width <= 680 && size.height <= 740,
        "Model selection stays inside the Settings content column")
    }
  }
}

private func speechSetupFixture(onlyUnavailable: Bool = false) throws -> ManagerSnapshot {
  let base = try endpointFixture()
  func artifact(
    _ id: String, _ caps: [String]?, installed: Bool = true,
    backend: Bool = true, qualification: String = "qualified"
  ) -> [String: Any] {
    var runtime: [String: Any] = ["qualification": qualification]
    if let caps { runtime["capability"] = caps }
    return [
      "id": id, "kind": "weights", "format": "gguf", "label": id,
      "installed": installed, "backend_supported": backend, "total_size": 1000,
      "runtime": runtime, "default": id == "chat",
    ]
  }
  let catalog = try ManagerWire.decode(
    ModelCatalog.self,
    from: JSONSerialization.data(withJSONObject: [
      "schema": 3,
      "models": [
        [
          "id": "speech-fixture", "display": "Speech fixture",
          "capability": ["chat", "transcription"], "installed": true, "total_size": 1000,
          "artifacts": onlyUnavailable
            ? [artifact("unqualified", ["transcription"], qualification: "unqualified")]
            : [
              artifact("chat", ["chat"]), artifact("asr", ["chat", "transcription"]),
              artifact("legacy", nil), artifact("empty", []), artifact("align", ["alignment"]),
              artifact("tts", ["audio_generation"]),
              artifact("cuda", ["transcription"], backend: false),
              artifact("unqualified", ["transcription"], qualification: "unqualified"),
              artifact("download", ["transcription"], installed: false),
            ],
        ]
      ],
    ]))
  return ManagerSnapshot(
    identity: base.identity, readiness: base.readiness,
    catalog: catalog, runners: base.runners, servers: base.servers ?? [])
}
