import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Structured Reads wire parity")
struct StructuredReadsTests {
  @Test func picturesUseTheWebReferenceAndDoNotLeakIntoQuestionSets() throws {
    let url = "data:image/png;base64,YQ=="
    #expect(ReadPicture.reference(url) == "q-g1g74vqtuh")
    #expect(ReadPicture.reference("🖼️ Swedish å") == "d-2fj4t9chyqx")
    var draft = ReadDraft()
    draft.images = [ReadPicture(name: "image.png", url: url)]
    draft.steps = 2
    draft.think = 512
    #expect(draft.setBody["images"] == nil)
    #expect(draft.request(model: "diffusion")["images"] == .array([.string(url)]))
    #expect(
      try draft.orderedJSON(model: "diffusion", includeImageData: false).contains(
        "<attached: image.png>"))
    let restored = try ReadDraft.parse(Data(draft.orderedJSON().utf8))
    #expect(restored.steps == 2 && restored.think == 512 && restored.images.isEmpty)
    #expect(throws: (any Error).self) {
      try ReadDraft.parse(Data(draft.orderedJSON(model: "diffusion").utf8))
    }
    #expect(throws: (any Error).self) {
      try ReadDraft.parse(
        Data(#"{"questions":{"q1":{"type":"noul","ask_if":{"q0":["yes"]}}}}"#.utf8))
    }
    let run = ConversationValue.object(["images": .array(draft.images.map(\.historyReference))])
    let table = ConversationValue.object([draft.images[0].ref: .string(url)])
    #expect(try ReadPicture.restore(run, table: table) == draft.images)
    #expect(throws: (any Error).self) { try ReadPicture.restore(run, table: .object([:])) }
    #expect(throws: (any Error).self) {
      try ReadPicture.restore(
        run, table: .object([draft.images[0].ref: .string("https://example.com/image.png")]))
    }
    draft.images = Array(repeating: draft.images[0], count: 17)
    #expect(draft.validation() != nil)
  }
  @Test func bundledExampleMatchesTheSharedRequestAndKeepsAuthoredOrder() throws {
    let draft = try ReadDraft.example
    #expect(draft.validation() == nil && draft.samples == 0)
    #expect(draft.state.hasPrefix("Subject: Portal down again\n\n"))
    #expect(draft.state.hasSuffix("\n\n- Dana, Ops lead at Northwind"))
    #expect(
      draft.questions.map(\.questionID) == [
        "need_action_within", "message_about", "upset_sender",
      ])
    #expect(draft.questions.map(\.kind) == [.noul, .choice, .score])
    #expect(draft.questions[0].yesMeans == "an outage or blocker affecting many people now")
    #expect(draft.questions[0].noMeans == "a request that can wait a day")
    #expect(draft.questions[1].options.map(\.name) == ["outage", "billing", "feature", "other"])
    #expect(draft.questions[2].levels.map(\.name) == ["calm", "annoyed", "furious"])
    for (index, question) in draft.questions.enumerated() {
      #expect(!question.idTouched)
      #expect(
        question.questionID
          == ReadQuestion.derivedID(
            question.instructions, taken: draft.questions.prefix(index).map(\.questionID)))
    }
    let again = try ReadDraft.parse(Data(draft.orderedJSON(model: "diffusion").utf8))
    #expect(again.request(model: "diffusion") == draft.request(model: "diffusion"))
    #expect(again.ordering == draft.ordering)
  }
  @Test func loadingTheExampleCreatesIndependentEditableRows() throws {
    var draft = try ReadDraft.example
    let another = try ReadDraft.example
    #expect(Set(draft.questions.map(\.id)).isDisjoint(with: another.questions.map(\.id)))
    draft.questions[1].options[0].name = "Changed"
    draft.questions[2].levels[0].name = "Changed"
    draft.state = "Changed"
    #expect(another.questions[1].options[0].name == "outage")
    #expect(another.questions[2].levels[0].name == "calm")
    #expect(try ReadDraft.example.request(model: "m") == another.request(model: "m"))
  }
  @Test func descriptiveIDsMatchWebRulesAndCollisions() {
    #expect(ReadQuestion.derivedID("Is the customer angry?") == "customer_angry")
    #expect(ReadQuestion.derivedID("What is this message about?") == "message_about")
    #expect(ReadQuestion.derivedID("", taken: ["q", "q_2"]) == "q_3")
    #expect(ReadQuestion.derivedID("Is this urgent?", taken: ["urgent"]) == "urgent_2")
  }
  @Test func orderedJSONRoundTripsQuestionsAndEscapedOptionsWithoutAlphabetizing() throws {
    let json =
      #"{"questions":{"z_last":{"type":"choice","instructions":"Team?","criteria":{"Zebra":"first","A \"quoted\" option":"second"}},"a_first":{"type":"noul","instructions":"OK?"}},"samples":2}"#
    let draft = try ReadDraft.parse(Data(json.utf8))
    #expect(draft.questions.map(\.questionID) == ["z_last", "a_first"])
    #expect(draft.questions[0].options.map(\.name) == ["Zebra", "A \"quoted\" option"])
    let again = try ReadDraft.parse(Data(draft.orderedJSON().utf8))
    #expect(again.questions.map(\.questionID) == draft.questions.map(\.questionID))
    #expect(again.questions[0].options.map(\.name) == draft.questions[0].options.map(\.name))
    #expect(again.setBody == draft.setBody)
    #expect(throws: (any Error).self) {
      try ReadDraft.parse(Data(#"{"q":{"type":"noul"},"q":{"type":"noul"}}"#.utf8))
    }
  }
  @Test func confidenceBandsAndNearTiesUseTheWebThresholds() throws {
    #expect(
      [0.49, 0.5, 0.69, 0.7, 0.89, 0.9].map(ReadResponse.Answer.confidenceBin) == [
        0, 1, 1, 2, 2, 3,
      ])
    let answer = try JSONDecoder().decode(
      ReadResponse.Answer.self,
      from: Data(
        #"{"type":"noul","noul":0.51,"confidence":0.02,"agreement":0.5,"outside":0.8}"#.utf8))
    #expect(answer.nearTie)
  }
  @Test func allQuestionKindsRoundTripWithoutChangingTheWire() throws {
    let data = Data(
      #"{"samples":"auto","state":"A ticket","questions":{"urgent":{"type":"boolean","instructions":"Urgent?","criteria":{"true":"Needs help","false":"No action"}},"team":{"type":"choice","instructions":"Which team?","criteria":{"Support":"Help","Sales":"Purchase"}},"priority":{"type":"score","instructions":"Priority?","criteria":["Low","Medium","High"]}}}"#
        .utf8)
    let draft = try ReadDraft.parse(data)
    #expect(draft.validation() == nil && draft.samples == 0 && draft.state == "A ticket")
    #expect(draft.request(model: "m")["questions"]?["urgent"]?["type"] == .string("noul"))
    #expect(draft.setBody["state"] == nil && draft.setBody["model"] == nil)
    let again = try ReadDraft.parse(Data(ReadDraft.json(draft.setBody).utf8))
    #expect(again.setBody == draft.setBody)
    #expect(
      again.questions.first { $0.kind == .score }?.levels.map(\.name) == ["Low", "Medium", "High"])
  }
  @Test func duplicateAndInvalidNamesNeverTrapAndNeverValidate() {
    var draft = ReadDraft()
    draft.questions.append(.init(questionID: "q1"))
    #expect(draft.validation() != nil)
    _ = draft.setBody
    draft.questions = [.init(questionID: "q1", kind: .choice)]
    draft.questions[0].options = [.init(name: "Same"), .init(name: " Same ")]
    #expect(draft.validation() != nil)
    _ = draft.setBody
    draft.questions[0].questionID = "two words"
    #expect(draft.validation() != nil)
  }
  @Test func malformedImportsAreAtomicFailures() {
    for text in [
      "[]", "null", #"{"q":{"type":"other"}}"#,
      #"{"questions":{"q":{"type":"score","criteria":[]}}}"#,
      #"{"samples":0,"questions":{"q":{"type":"noul"}}}"#,
      #"{"samples":33,"questions":{"q":{"type":"noul"}}}"#,
    ] {
      #expect(throws: (any Error).self) { try ReadDraft.parse(Data(text.utf8)) }
    }
  }
  @Test func capsAndTypeSwitchRetainDrafts() {
    var draft = ReadDraft()
    draft.samples = 8
    #expect(draft.validation(maxSamples: 4) != nil)
    draft.questions[0].options = [.init(name: "A"), .init(name: "B")]
    draft.questions[0].kind = .score
    draft.questions[0].kind = .choice
    #expect(draft.questions[0].options.map(\.name) == ["A", "B"])
  }
  @Test func confidenceIsTheRunnerMeasureNotWinningProbability() throws {
    let answer = try JSONDecoder().decode(
      ReadResponse.Answer.self,
      from: Data(
        #"{"type":"noul","noul":0.62,"confidence":0.24,"agreement":0.75,"outside":0.9}"#.utf8))
    #expect(answer.confidence == 0.24 && answer.label == "Yes")
    #expect(answer.bars.map(\.probability) == [0.62, 0.38])
    #expect(answer.outside == 0.9)  // Not renormalized into the label bars.
  }
  @Test func scoreOrderIsNumericAndNotAlphabetical() throws {
    let answer = try JSONDecoder().decode(
      ReadResponse.Answer.self,
      from: Data(
        #"{"type":"score","score":0.7,"level":"Medium","confidence":0.4,"agreement":1,"outside":0,"legend":{"10":"High","2":"Medium","1":"Low"},"probabilities":{"1":0.2,"2":0.6,"10":0.2}}"#
          .utf8))
    #expect(answer.bars.map(\.name) == ["Low", "Medium", "High"])
  }
}
