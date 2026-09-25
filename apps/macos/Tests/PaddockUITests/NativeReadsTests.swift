import Foundation
import PaddockClient
import PaddockConversationCore
import Testing

@testable import PaddockUI

@Suite("Native Reads state", .serialized) @MainActor
struct NativeReadsTests {
  @Test func visionReadsRetainPixelsAndStepsInSharedHistory() async throws {
    let m = model()
    let originalAPI = m.api
    m.api = { path, method, body, query in
      if path.hasSuffix("1234/server") {
        return try value(
          #"{"structured_read":{"canvas_width":256,"images":true,"max_steps":8,"think":true}}"#)
      }
      return try await originalAPI(path, method, body, query)
    }
    await m.refresh()
    let picture = ReadPicture(name: "picture.png", url: "data:image/png;base64,YQ==")
    m.draft.images = [picture]
    m.draft.steps = 2
    m.draft.think = 128
    #expect(m.canRun && m.hasWork && m.current?.maxSteps == 8)
    m.beginJSON()
    #expect(m.applyJSON(m.jsonText) && m.draft.images == [picture])
    var saved: ConversationValue?
    m.api = { path, method, body, _ in
      if path == "api/runners/1234/v1/systemone" {
        #expect(method == "POST" && body?["images"] == .array([.string(picture.url)]))
        #expect(body?["steps"] == .number(2) && body?["think"] == .number(128))
        return try value(
          #"{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.99,"confidence":0.98,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":256,"questions":[],"timing":{"total_ms":12}}}"#
        )
      }
      if path.hasPrefix("api/read-history/") {
        if method == "PUT" {
          saved = try JSONDecoder().decode(
            ConversationValue.self, from: Data(body!["doc"]!.string!.utf8))
          return .object(["read": .object(["revision": .string("saved")])])
        }
        return .object(["revision": .string("saved"), "doc": .string(try ReadDraft.json(saved!))])
      }
      return .array([])
    }
    m.run()
    await m.settle()
    #expect(m.error == nil && m.historyError == nil && !m.unsavedRead)
    #expect(saved?["images"]?[picture.ref] == .string(picture.url))
    #expect(saved?["runs"]?.array?.first?["images"] == .array([picture.historyReference]))
    let id = try #require(m.activeSession?.id)
    m.reset()
    await m.openSession(id)
    #expect(m.draft.images == [picture] && m.draft.steps == 2 && m.draft.think == 128)
    #expect(!m.unsavedRead && !m.stale)
    m.draft.images = []
    #expect(m.unsavedRead && m.stale)
  }
  @Test func textOnlyReaderNeverSilentlyDiscardsPictures() async {
    let m = model()
    await m.refresh()
    m.draft.images = [ReadPicture(name: "picture.png", url: "data:image/png;base64,YQ==")]
    #expect(!m.canRun && m.validation != nil)
    m.routeError("images: the image is too large")
    #expect(m.stateError != nil)
  }
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
      if path == "api/read-history" { return .array([]) }
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
  @Test func exampleRunsAllThreeQuestionsOnTheSelectedReaderAndRetainsHistory() async throws {
    let m = model()
    await m.refresh()
    let example = try ReadDraft.example
    var requests = 0
    m.api = { path, method, body, _ in
      if path == "api/read-history" { return .array([]) }
      if path.hasPrefix("api/read-history/") {
        return .object(["read": .object(["revision": .string("saved")])])
      }
      requests += 1
      #expect(path == "api/runners/1234/v1/systemone" && method == "POST")
      #expect(body == example.request(model: "diffusion"))
      return try value(
        #"{"model":"diffusion","answers":{"need_action_within":{"type":"noul","noul":0.99,"confidence":0.98,"agreement":1,"outside":0.01},"message_about":{"type":"choice","choice":"outage","probabilities":{"outage":0.97,"billing":0.01,"feature":0.01,"other":0.01},"confidence":0.96,"agreement":1,"outside":0.01},"upset_sender":{"type":"score","score":1.9,"level":"furious","legend":{"0":"calm","1":"annoyed","2":"furious"},"probabilities":{"0":0,"1":0.1,"2":0.9},"confidence":0.85,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":256,"questions":[],"timing":{"total_ms":12}}}"#
      )
    }
    m.draft.state = "Draft replaced only after the view's confirmation"
    m.jsonText = "{unfinished"
    m.fileName = "previous.txt"
    m.runExample()
    m.runExample()  // A second click during the request cannot start another run.
    await m.settle()
    #expect(requests == 1 && !m.busy && !m.stale)
    #expect(m.error == nil && m.questionsError == nil && m.historyError == nil)
    #expect(m.result?.response.answers.count == 3 && m.activeSession?.runs.count == 1)
    #expect(m.result?.state == example.state && m.fileName.isEmpty)
    let expectedJSON = try example.orderedJSON()
    #expect(!m.hasUnappliedJSON && m.jsonText == expectedJSON)
    #expect(m.draft.ordering == example.ordering)
    m.editInstructions(m.draft.questions[0].id, text: "Is the customer angry?")
    #expect(m.draft.questions[0].questionID == "customer_angry")
  }
  @Test func exampleWithoutAReaderNeverDiscardsTheDraft() async {
    let m = model()
    m.draft.state = "Keep this"
    m.jsonText = "{unfinished"
    let before = m.draft
    m.runExample()
    await m.settle()
    #expect(m.draft == before && m.jsonText == "{unfinished" && m.result == nil)
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
      if path == "api/read-history" { return .array([]) }
      if path.hasPrefix("api/read-history/") {
        return .object(["read": .object(["revision": .string("saved")])])
      }
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
      if path == "api/read-history" { return .array([]) }
      if path.hasPrefix("api/read-history/") {
        return .object(["read": .object(["revision": .string("saved")])])
      }
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

  @Test func historyUsesTheSharedWebDocumentAndReopensTheFullInput() async throws {
    let m = model()
    await m.refresh()
    m.draft.state = "The complete input that can be rerun"
    m.fileName = "source.txt"
    var saved: String?
    let summary: ConversationValue = .object([
      "id": .string("session"), "title": .string("Source"),
      "model": .string("diffusion"), "runs": .number(1), "updatedAt": .number(1000),
    ])
    m.api = { path, method, body, query in
      if path == "api/read-history" { return .array([summary]) }
      if path.hasPrefix("api/read-history/") {
        #expect(method == "PUT" && query["revision"] == "")
        saved = body?["doc"]?.string
        return .object(["read": .object(["revision": .string("revision-1")])])
      }
      return try value(
        #"{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.9,"confidence":0.8,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":16,"questions":[],"timing":{"total_ms":1}}}"#
      )
    }
    m.run()
    await m.settle()
    let text = try #require(saved)
    let doc = try ReadHistoryDocument(json: text)
    #expect(doc.runs.first?["state"] == .string(m.draft.state))
    #expect(doc.runs.first?["fileName"] == .string("source.txt"))
    #expect(!m.historyUnsaved && m.historyError == nil)
    let restored = model()
    let api = restored.api
    restored.api = { path, method, body, query in
      if path == "api/read-history" { return .array([summary]) }
      if path.hasPrefix("api/read-history/") {
        if method == "DELETE" {
          #expect(query["revision"] == "revision-1")
          return .null
        }
        return .object(["doc": .string(text), "revision": .string("revision-1")])
      }
      return try await api(path, method, body, query)
    }
    await restored.refresh()
    await restored.openSession(doc.id)
    #expect(restored.runs.count == 1 && restored.result?.id == m.result?.id)
    #expect(restored.draft.state == m.draft.state && restored.fileName == "source.txt")
    #expect(
      restored.historyError == nil && restored.result!.at.timeIntervalSince1970 > 1_700_000_000)
    await restored.clearHistory()
    #expect(restored.runs.isEmpty)
  }

  @Test func failedHistorySaveKeepsAnswersAndDraftForRetry() async throws {
    let m = model()
    await m.refresh()
    m.draft.state = "Preserve me"
    m.api = { path, _, _, _ in
      if path.hasPrefix("api/read-history/") { throw ConversationFailure.http(409) }
      return try value(
        #"{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.9,"confidence":0.8,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":16,"questions":[],"timing":{"total_ms":1}}}"#
      )
    }
    m.run()
    await m.settle()
    #expect(m.historyUnsaved && m.historyError != nil && m.result != nil)
    #expect(m.draft.state == "Preserve me" && m.activeSession != nil)
  }

  @Test func lateHistoryOpenCannotOverwriteANewerDraft() async throws {
    let m = model()
    m.api = { _, _, _, _ in
      m.draft.state = "Newer input"
      return .object([
        "doc": .string(#"{"id":"session","title":"Old","runs":[]}"#),
        "revision": .string("r"),
      ])
    }
    await m.openSession("session")
    #expect(m.draft.state == "Newer input" && m.activeSession == nil && !m.openingSession)
  }

  @Test func sidebarSearchAndOrderingMatchWebHistoryAndRecoverAfterFailure() async throws {
    let m = model()
    let rows = try value(
      #"[{"id":"old","title":"Older","model":"diffusion","runs":3,"updatedAt":1},{"id":"new","title":"Café ticket","model":"diffusion","runs":2,"updatedAt":3},{"id":"middle","title":"Middle","model":"diffusion","runs":1,"updatedAt":2}]"#
    )
    m.api = { _, _, _, _ in rows }
    await m.refreshHistory()
    #expect(m.historyLoaded && m.visibleSessions(search: "").map(\.id) == ["new", "middle", "old"])
    #expect(m.visibleSessions(search: "  CAFÉ ").map(\.id) == ["new"])
    #expect(m.visibleSessions(search: "absent").isEmpty)
    m.api = { _, _, _, _ in throw ConversationFailure.http(503) }
    await m.refreshHistory()
    #expect(m.historyListError != nil && m.sessions.count == 3)
    m.api = { _, _, _, _ in rows }
    await m.refreshHistory()
    #expect(m.historyListError == nil && m.historyLoaded)
  }

  func savedRead(_ id: String, title: String = "Ticket") throws -> String {
    var doc = try value(
      #"{"id":"read","title":"Ticket","model":"diffusion","createdAt":1000,"updatedAt":2000,"runs":[{"id":"73D49A22-9423-4601-95D9-6F855224A452","at":2000,"port":1234,"state":"The complete ticket","fileName":"ticket.txt","questions":{"q1":{"type":"noul","instructions":"Urgent?"}},"samples":"auto","response":{"model":"diffusion","answers":{"q1":{"type":"noul","noul":0.9,"confidence":0.8,"agreement":1,"outside":0.01}},"diagnostics":{"reads":1,"canvas":256,"questions":[],"timing":{"total_ms":12}}},"ms":15}]}"#
    ).object!
    doc["id"] = .string(id)
    doc["title"] = .string(title)
    return try ReadDraft.json(.object(doc))
  }

  @Test func sidebarOpensWithoutARunningReaderAndOnlyPromptsForActualEdits() async throws {
    let m = model()
    let doc = try savedRead("one")
    m.api = { _, _, _, _ in .object(["doc": .string(doc), "revision": .string("r1")]) }
    await m.openSession("one")
    #expect(m.current == nil && m.result?.state == "The complete ticket")
    #expect(m.activeSession?.id == "one" && !m.unsavedRead && !m.historyNavigationBlocked)
    m.draft.state += " edited"
    #expect(m.unsavedRead)
    m.draft.state = "The complete ticket"
    #expect(!m.unsavedRead)
    m.jsonText = "{unfinished"
    #expect(m.unsavedRead)
  }

  @Test func sidebarRenamePreservesAllRunsAndDraftWhileUpdatingTheRevision() async throws {
    let m = model()
    let original = try savedRead("one")
    m.api = { _, _, _, _ in .object(["doc": .string(original), "revision": .string("r1")]) }
    await m.openSession("one")
    m.draft.state = "Newer unsent input"
    var writes = 0
    m.api = { _, method, body, query in
      if method == "GET" { return .object(["doc": .string(original), "revision": .string("r1")]) }
      #expect(query["revision"] == (method == "PUT" ? "r1" : "r2"))
      if method == "PUT" {
        writes += 1
        let renamed = try ReadHistoryDocument(json: #require(body?["doc"]?.string))
        let before = try ReadHistoryDocument(json: original)
        #expect(
          renamed.runs == before.runs && renamed.value["updatedAt"] == before.value["updatedAt"])
        #expect(renamed.value["title"] == .string("Renamed"))
        return .object(["read": .object(["revision": .string("r2")])])
      }
      #expect(method == "DELETE")
      return .null
    }
    await m.renameSession("one", title: " Renamed ")
    #expect(writes == 1 && m.activeSession?.value["title"] == .string("Renamed"))
    #expect(m.draft.state == "Newer unsent input" && m.runs.count == 1 && m.historyError == nil)
    await m.removeSession("one")
    #expect(m.activeSession == nil && m.runs.isEmpty && m.draft.state.isEmpty)
  }

  @Test func sidebarCanDeleteAnUnopenedReadWithoutChangingTheOpenRead() async throws {
    let m = model()
    let one = try savedRead("one")
    let two = try savedRead("two")
    m.api = { path, method, _, query in
      if method == "DELETE" {
        #expect(path == "api/read-history/two" && query["revision"] == "two-revision")
        return .null
      }
      let isTwo = path.hasSuffix("/two")
      return .object([
        "doc": .string(isTwo ? two : one),
        "revision": .string(isTwo ? "two-revision" : "one-revision"),
      ])
    }
    await m.openSession("one")
    let before = m.draft
    await m.removeSession("two")
    #expect(m.activeSession?.id == "one" && m.draft == before && m.runs.count == 1)
  }

  @Test func sidebarRevisionConflictsRetainTheReadAndItsInput() async throws {
    let m = model()
    let original = try savedRead("one")
    m.api = { _, _, _, _ in .object(["doc": .string(original), "revision": .string("r1")]) }
    await m.openSession("one")
    let before = m.draft
    m.api = { _, method, _, _ in
      if method == "GET" { return .object(["doc": .string(original), "revision": .string("r1")]) }
      throw ConversationFailure.http(409)
    }
    await m.renameSession("one", title: "Not saved")
    #expect(m.historyError != nil && m.activeSession?.value["title"] == .string("Ticket"))
    await m.removeSession("one")
    #expect(m.historyError != nil && m.activeSession?.id == "one" && m.draft == before)
    #expect(!m.saving && m.runs.count == 1)
  }

  @Test func staleListRefreshCannotResurrectADeletedRead() async throws {
    let m = model()
    let doc = try savedRead("one")
    let rows = try value(
      #"[{"id":"one","title":"Ticket","model":"diffusion","runs":1,"updatedAt":2000}]"#)
    m.api = { _, _, _, _ in rows }
    await m.refreshHistory()
    m.api = { path, method, _, _ in
      if path == "api/read-history" {
        await m.removeSession("one")
        return rows  // This GET started before the DELETE was acknowledged.
      }
      if method == "GET" { return .object(["doc": .string(doc), "revision": .string("r1")]) }
      return .null
    }
    await m.refreshHistory()
    #expect(m.sessions.isEmpty && m.historyError == nil)
  }

  @Test func renameCannotAdoptAnExternallyChangedReadOverTheOpenSnapshot() async throws {
    let m = model()
    let original = try savedRead("one")
    m.api = { _, _, _, _ in .object(["doc": .string(original), "revision": .string("r1")]) }
    await m.openSession("one")
    m.api = { _, method, _, _ in
      #expect(method == "GET")
      return .object(["doc": .string(original), "revision": .string("r2")])
    }
    await m.renameSession("one", title: "Changed")
    #expect(m.historyError != nil && m.activeSession?.value["title"] == .string("Ticket"))
  }
}
