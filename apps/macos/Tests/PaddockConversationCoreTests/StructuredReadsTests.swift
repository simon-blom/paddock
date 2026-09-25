import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Structured Reads wire parity")
struct StructuredReadsTests {
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
