import Foundation
import PaddockClient
import PaddockConversationCore
import Testing

@testable import PaddockUI

@Suite("Native Reads state", .serialized) @MainActor
struct NativeReadsTests {
  func value(_ json: String) throws -> ConversationValue {
    try JSONDecoder().decode(ConversationValue.self, from: Data(json.utf8))
  }
  func model() -> NativeReadsModel {
    let m = NativeReadsModel(client: NativeManager())
    m.api = { path, _, _, _ in
      if path == "api/runners" {
        return try value(
          #"[{"port":1234,"model":"diffusion","vendor":"Google"},{"port":1235,"model":"chat-only"}]"#
        )
      }
      if path == "api/runners/1234/server" {
        return try value(
          #"{"structured_read":{"canvas_width":256,"max_questions":64,"max_samples":32,"types":["noul","choice","score"]}}"#
        )
      }
      if path == "api/reads" { return .array([]) }
      if path.hasPrefix("api/read-runs/") { return .array([]) }
      return .object([:])
    }
    return m
  }
  @Test func capabilityDiscoveryNeverGuessesFromModelNames() async {
    let m = model()
    await m.refresh()
    #expect(m.readers.count == 1 && m.port == 1234 && !m.canRun)
    m.draft.state = "The sky is blue."
    #expect(m.canRun)
  }
  @Test func invalidJsonKeepsTheCompleteEditor() {
    let m = model()
    m.draft.state = "Keep this"
    let before = m.draft
    m.applyJSON("{invalid")
    #expect(m.draft == before && m.questionsError != nil)
    m.applyJSON(#"{"q":{"type":"noul","instructions":"Question?"}}"#)
    #expect(m.draft.state == "Keep this" && m.questionsError == nil)
  }
  @Test func savedSetFailureDoesNotHideTheLiveReaderAndRecoveryClearsOnlyItsError() async {
    let m = model()
    let api = m.api
    m.api = { path, method, body, query in
      if path == "api/reads" { throw ConversationFailure.http(503) }
      return try await api(path, method, body, query)
    }
    await m.refresh()
    #expect(m.readers.count == 1 && m.error != nil)
    m.api = api
    await m.refresh()
    #expect(m.error == nil && m.readers.count == 1)
    m.applyJSON("{unfinished")
    let error = m.questionsError
    await m.refresh()
    #expect(m.questionsError == error && error != nil)
  }
  @Test func unappliedJsonProtectsQuitAndCannotSilentlySaveOtherQuestions() async {
    let m = model()
    m.beginJSON()
    #expect(!m.dirty)
    m.jsonText = "{unfinished"
    #expect(m.hasUnappliedJSON && m.hasWork)
    m.beginJSON()
    #expect(m.jsonText == "{unfinished")
    m.setName = "Set"
    await m.save()
    #expect(m.selectedSet == nil && m.error != nil)
  }
  @Test func revisionConflictKeepsDraftAndDoesNotPublishFalseSuccess() async {
    let m = model()
    m.setName = "Triage"
    var revision: ConversationValue?
    m.api = { _, _, body, _ in
      revision = body?["revision"]
      throw ConversationFailure.http(409)
    }
    await m.save()
    #expect(revision == .string("") && m.selectedSet == nil && m.dirty && m.setName == "Triage")
    #expect(m.error != nil && !m.saving)
  }
  @Test func failedDeletionRetainsReviewedSetAndRevision() async throws {
    let m = model()
    let set = NativeReadsModel.SavedSet(
      id: "set-1", name: "Triage", body: try ReadDraft.json(m.draft.setBody), revision: "old")
    m.open(set)
    var query: [String: String] = [:]
    m.api = { _, _, _, q in
      query = q
      throw ConversationFailure.http(409)
    }
    await m.remove()
    #expect(query["revision"] == "old" && m.selectedSet == set && m.error != nil)
  }
  @Test func runResultsRetainSubmittedQuestionsAndBecomeStaleAfterEdits() async throws {
    let m = model()
    await m.refresh()
    m.draft.state = "Sky is blue."
    m.api = { path, _, body, _ in
      if path.hasPrefix("api/read-runs/") { return .null }
      #expect(body?["model"] == .string("diffusion") && body?["samples"] == .string("auto"))
      return try value(
        #"{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.9,"confidence":0.8,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":16,"questions":[{"id":"q1","label":"yes","entropy":0.1,"label_mass":0.99}],"timing":{"total_ms":12}}}"#
      )
    }
    m.run()
    await m.settle()
    #expect(m.result != nil && !m.stale && !m.busy)
    m.draft.questions[0].instructions = "Changed"
    #expect(m.stale && m.result?.questions[0].instructions == "")
  }
  @Test func cancellationDoesNotPublishLateAnswers() async {
    let m = model()
    await m.refresh()
    m.draft.state = "Text"
    m.api = { _, _, _, _ in
      try await Task.sleep(for: .seconds(30))
      return .null
    }
    m.run()
    m.cancel()
    await m.settle()
    #expect(!m.busy && m.result == nil && m.error == nil)
  }
  @Test func savingDoesNotOverwriteEditsMadeWhileAwaitingTheServer() async throws {
    let m = model()
    m.setName = "Submitted"
    m.api = { _, _, body, _ in
      m.setName = "Newer edit"
      m.draft.questions[0].instructions = "Keep my newer question"
      return .object([
        "set": .object([
          "id": body!["id"]!, "name": body!["name"]!,
          "body": body!["body"]!, "revision": .string("saved"),
        ])
      ])
    }
    await m.save()
    #expect(m.selectedSet?.name == "Submitted" && m.setName == "Newer edit")
    #expect(m.draft.questions[0].instructions == "Keep my newer question" && m.dirty)
  }
  @Test func historyUsesAGlobalByteLimitAndReleasesTheCurrentSnapshot() async throws {
    let m = model()
    await m.refresh()
    m.draft.state = "Document"
    m.api = { _, _, _, _ in
      try value(
        #"{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.9,"confidence":0.8,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":16,"questions":[],"timing":{"total_ms":1}}}"#
      )
    }
    m.run()
    await m.settle()
    #expect(m.result != nil)
    m.trimHistory(maxBytes: 1)
    #expect(m.result == nil && m.runs.isEmpty)
  }
  @Test func navigationKeepsDraftAndQuitProtectsIt() {
    let workspace = WorkspaceModel()
    workspace.reads.draft.state = "Document"
    workspace.navigation.studio = .reads
    workspace.navigation.showManager(.runners)
    #expect(workspace.studioNeedsQuitConfirmation)
    workspace.navigation.mode = .studio
    #expect(workspace.navigation.studio == .reads && workspace.reads.draft.state == "Document")
  }

  @Test func questionIDsFollowInstructionsUntilOverriddenAndDuplicatesStayAdjacent() {
    let m = model()
    let id = m.draft.questions[0].id
    m.editInstructions(id, text: "Is the customer angry?")
    #expect(m.draft.questions[0].questionID == "customer_angry")
    m.editID(id, text: "my: decision")
    m.editInstructions(id, text: "Is this urgent?")
    #expect(m.draft.questions[0].questionID == "my_decision")
    m.editID(id, text: "")
    #expect(m.draft.questions[0].questionID == "urgent")
    m.add(.noul)
    m.duplicate(id)
    #expect(m.draft.questions.map(\.questionID) == ["urgent", "urgent_2", "q1"])
  }

  @Test func jsonRunAppliesTheBufferAndInvalidJsonNeverSendsOtherQuestions() async throws {
    let m = model()
    await m.refresh()
    m.draft.state = "Keep this document"
    m.beginJSON()
    var requests = 0
    m.api = { path, _, body, _ in
      if path.hasPrefix("api/read-runs/") { return .null }
      requests += 1
      #expect(body?["questions"]?["urgent"]?["type"] == .string("noul"))
      return try value(
        #"{"model":"diffusion","answers":{"urgent":{"type":"noul","noul":0.5,"confidence":0,"agreement":0.5,"outside":0.1}},"diagnostics":{"reads":2,"canvas":16,"questions":[],"timing":{"total_ms":1}}}"#
      )
    }
    m.jsonText = "{unfinished"
    m.run(applyJSON: true)
    await m.settle()
    #expect(requests == 0 && m.questionsError != nil)
    m.jsonText = #"{"urgent":{"type":"noul","instructions":"Urgent?"}}"#
    m.run(applyJSON: true)
    await m.settle()
    #expect(requests == 1 && m.result?.questions[0].questionID == "urgent")
    #expect(m.draft.state == "Keep this document" && m.questionsError == nil && !m.stale)
  }

  @Test func runnerErrorsLandAtTheirFieldsAndUnknownQuestionsStayVisible() {
    let m = model()
    m.routeError(#"question "q1": no usable label token"#)
    #expect(m.rowErrors["q1"] != nil && m.error == nil)
    m.routeError("state: exceeds context")
    #expect(m.stateError != nil)
    m.routeError("samples: expected at most 4")
    #expect(m.questionsError != nil)
    m.routeError(#"question "deleted": invalid"#)
    #expect(m.error != nil)
    m.clearRunErrors()
    #expect(m.rowErrors.isEmpty && m.stateError == nil && m.questionsError == nil)
  }

  @Test func historyRestoresAcrossModelsWithoutRetainingTheFullSourceDocument() async throws {
    let m = model()
    await m.refresh()
    m.draft.state = String(repeating: "private document ", count: 200) + "SENSITIVE_END"
    var saved: ConversationValue?
    m.api = { path, method, body, _ in
      if path.hasPrefix("api/read-runs/") {
        if method == "POST" { saved = body }
        return .null
      }
      return try value(
        #"{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.9,"confidence":0.8,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":16,"questions":[],"timing":{"total_ms":1}}}"#
      )
    }
    m.run()
    await m.settle()
    let record = try #require(saved)
    #expect(record["state"] == nil && record["request"] == nil)
    #expect(
      !String(decoding: try JSONEncoder().encode(record), as: UTF8.self).contains("SENSITIVE_END"))
    #expect((record["excerpt"]?.string?.count ?? 1000) <= 120)
    let restored = model()
    let api = restored.api
    restored.api = { path, method, body, query in
      if path == "api/read-runs/draft" { return method == "GET" ? .array([record]) : .null }
      return try await api(path, method, body, query)
    }
    await restored.refresh()
    #expect(restored.runs.count == 1 && restored.result?.id == m.result?.id)
    #expect(restored.previousRead && restored.draft.state.isEmpty && restored.historyError == nil)
    await restored.clearHistory()
    #expect(restored.runs.isEmpty)
  }
}
