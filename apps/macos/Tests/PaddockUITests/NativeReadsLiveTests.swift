import Foundation
import PaddockConversationCore
import Testing

@testable import PaddockClient
@testable import PaddockUI

/// Opt-in qualification against an already running local model. Isolated DB,
/// no model launch/stop, windows, user's conversation writes or printed keys.
@Suite("Live native Reads", .serialized) @MainActor
struct NativeReadsLiveTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_READS_LIVE"] == "1"))
  func allQuestionTypesJSONSaveHistoryAndReopen() async throws {
    let environment = ProcessInfo.processInfo.environment
    let root = try #require(environment["PADDOCK_DATA"])
    #expect(root.hasPrefix("/tmp/paddock-reads-live."))
    guard root.hasPrefix("/tmp/paddock-reads-live.") else { return }
    let library = URL(fileURLWithPath: try #require(environment["PADDOCK_READS_LIBRARY"]))
    let client = NativeManager(libraryURL: library)
    let model = NativeReadsModel(client: client)
    await model.refresh()
    #expect(model.error == nil)
    _ = try #require(
      model.current, "No running structured-read model discovered in isolated configuration")
    let example = try ReadDraft.example
    model.draft = example
    model.draft.samples = 1
    model.setName = "Native Reads qualification"
    await model.save()
    let saved = try #require(model.selectedSet)
    model.run()
    await model.settle()
    #expect(model.error == nil && model.historyError == nil && model.questionsError == nil)
    let first = try #require(model.result)
    #expect(first.response.answers.count == 3)
    #expect(first.response.answers["need_action_within"]?.noul ?? 0 > 0.5)
    #expect(first.response.answers["message_about"]?.choice == "outage")
    #expect(first.response.answers["upset_sender"]?.score ?? 0 > 1)
    model.beginJSON()
    var repeated = model.draft
    repeated.samples = 3
    model.jsonText = try repeated.orderedJSON()
    model.run(applyJSON: true)
    await model.settle()
    let second = try #require(model.result)
    let session = try #require(model.activeSession?.id)
    #expect(second.id != first.id && second.response.diagnostics.reads == 3)
    #expect(second.response.diagnostics.questions.allSatisfy { $0.reads?.count == 3 })
    await client.close()
    let reopened = NativeManager(libraryURL: library)
    let restored = NativeReadsModel(client: reopened)
    await restored.refresh()
    let set = try #require(restored.sets.first { $0.id == saved.id })
    restored.open(set)
    await restored.openSession(session)
    #expect(restored.runs.count == 2 && restored.draft.state == model.draft.state)
    #expect(
      restored.draft.questions.map(\.questionID) == example.questions.map(\.questionID))
    await restored.clearHistory()
    restored.open(set)
    await restored.remove()
    #expect(restored.error == nil)
    await reopened.close()
  }
}
