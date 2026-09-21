import AppKit
import Foundation
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Speech model lifecycle parity", .serialized) @MainActor
struct StudioSpeechModelsTests {
  @Test func startUsesSavedRevisionAndNeverCreatesOrRecords() async throws {
    let client = SpeechFixture()
    let model = StudioSpeechModels(client: client)
    let row = try speechRow()
    #expect(await client.commands.isEmpty)
    model.act(row, start: true)
    model.act(row, start: true)
    #expect(model.busy && model.port == 11542)
    await model.settle()
    #expect(!model.busy && model.error == nil)
    #expect(await client.commands == ["start:11542:saved-revision:false", "poll:7"])
  }

  @Test func stopUsesCurrentPidAndRetainsTheSavedEndpoint() async throws {
    let client = SpeechFixture(running: true)
    let model = StudioSpeechModels(client: client)
    model.act(try speechRow(running: true), start: false)
    await model.settle()
    #expect(await client.commands == ["stop:11542:42", "poll:7"])
    #expect(!model.busy && model.error == nil)
  }

  @Test func unknownOrChangedModelsAndNetworkStartsAreNotSilentlyAccepted() async throws {
    for client in [
      SpeechFixture(speech: false), SpeechFixture(modelID: "replacement"),
      SpeechFixture(local: false),
    ] {
      let model = StudioSpeechModels(client: client)
      model.act(try speechRow(), start: true)
      await model.settle()
      #expect(await client.commands.isEmpty)
      #expect(model.error != nil && !model.busy)
    }
  }

  @Test func failureStaysVisibleAndLostStatusNeverRepeatsTheMutation() async throws {
    let client = SpeechFixture()
    await client.setPollFailure(true)
    let model = StudioSpeechModels(client: client)
    let row = try speechRow()
    model.act(row, start: true)
    await model.settle()
    #expect(model.busy && model.pending?.id == 7 && model.error != nil)
    model.act(row, start: true)
    await client.setPollFailure(false)
    model.retryStatus()
    await model.settle()
    #expect(await client.commands.filter { $0.hasPrefix("start:") }.count == 1)
    #expect(!model.busy && model.error == nil)
    let failed = SpeechFixture(failJob: true)
    let other = StudioSpeechModels(client: failed)
    other.act(row, start: true)
    await other.settle()
    #expect(other.error == "Model weights are missing" && !other.busy)
  }

  @Test func unavailableActionsDoNotSubmitAndRowsFitBothAppearancesOffscreen() throws {
    _ = NSApplication.shared
    var calls = 0
    for count in [1, 4, 12] {
      let rows = try (0..<count).map {
        try speechRow(port: 11542 + $0, running: $0.isMultiple(of: 2))
      }
      for dark in [false, true] {
        let host = NSHostingController(
          rootView:
            StudioSpeechModelRows(rows: rows) { _, _ in calls += 1 }
            .frame(width: 344).environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: CGSize(width: 344, height: 600))
        #expect(size.width <= 344 && size.height <= 208)
      }
    }
    #expect(calls == 0)
    let client = SpeechFixture()
    let model = StudioSpeechModels(client: client)
    model.act(try speechRow(), start: false)
    model.act(try speechRow(running: true), start: true)
    model.canAct = { false }
    model.act(try speechRow(), start: true)
    #expect(!model.busy)
  }
}

private func speechRow(port: Int = 11542, running: Bool = false) throws
  -> StudioState.Audio.SpeechModel
{
  try JSONDecoder().decode(
    StudioState.Audio.SpeechModel.self,
    from: JSONSerialization.data(withJSONObject: [
      "port": port, "model": "whisper", "title": "Whisper Large V3 · Long speech model name",
      "vendor": "OpenAI", "running": running, "busy": false,
      "status": running ? "running · port \(port)" : "port \(port)", "canStart": !running,
      "canStop": running,
    ]))
}

private actor SpeechFixture: ManagerLoading {
  var commands: [String] = []
  var pollFailure = false
  let running: Bool
  let speech: Bool
  let modelID: String
  let local: Bool
  let failJob: Bool
  var action = "start"
  init(
    running: Bool = false, speech: Bool = true, modelID: String = "whisper", local: Bool = true,
    failJob: Bool = false
  ) {
    self.running = running
    self.speech = speech
    self.modelID = modelID
    self.local = local
    self.failJob = failJob
  }
  func setPollFailure(_ value: Bool) { pollFailure = value }
  func snapshot() async throws -> ManagerSnapshot {
    let base = try fixture()
    let servers = try ManagerWire.decode(
      [ConfiguredEndpoint].self,
      from: JSONSerialization.data(withJSONObject: [
        [
          "port": 11542, "model": modelID, "display": "Whisper", "running": running,
          "revision": "saved-revision", "local_only": local,
          "capability": speech ? ["transcription"] : ["chat"],
        ]
      ]))
    let runners = try ManagerWire.decode(
      [RunnerInfo].self,
      from: JSONSerialization.data(
        withJSONObject: running
          ? [
            [
              "port": 11542, "pid": 42, "status": "ok", "asr": "whisper",
              "endpoint": "http://127.0.0.1:11542",
            ]
          ] : []))
    return ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: base.catalog,
      runners: runners, servers: servers)
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    let state: String
    switch command {
    case .start(let port, let revision, let allowNetwork):
      commands.append("start:\(port):\(revision):\(allowNetwork)")
      action = "start"
      state = "running"
    case .stop(let port, let pid):
      commands.append("stop:\(port):\(pid)")
      action = "stop"
      state = "running"
    case .poll(let id):
      commands.append("poll:\(id)")
      if pollFailure { throw ManagerError.core("Synthetic status transport failure") }
      state = failJob ? "failed" : "succeeded"
    default: throw ManagerError.core("Unexpected lifecycle command")
    }
    return try ManagerWire.decode(
      ManagementJob.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": 7, "port": 11542, "action": action, "state": state,
        "message": failJob ? "Model weights are missing" : "Completed",
      ]))
  }
}
