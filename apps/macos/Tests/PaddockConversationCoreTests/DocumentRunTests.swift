import Foundation
import Testing

@testable import PaddockConversationCore

struct DocumentRunTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  static func terminal(_ text: String, status: String = "completed", usage: O? = nil) -> O {
    let message: V = .object([
      "type": .string("message"),
      "content": .array([
        .object(["type": .string("output_text"), "text": .string(text)])
      ]),
    ])
    return [
      "type": .string("response.\(status)"), "sequence_number": .number(2),
      "response": .object([
        "status": .string(status), "output": .array([message]),
        "usage": .object(usage ?? ["input_tokens": .number(10), "output_tokens": .number(5)]),
      ]),
    ]
  }
  func state(_ count: Int = 2) -> NativeDocumentRunState {
    NativeDocumentRunState(
      sourceID: "source",
      pages: (0..<count).map {
        ["page": .number(Decimal($0 + 7)), "state": .string("queued"), "text": .string("")]
      })
  }
  @Test func eachPageRestartsSequenceAndTerminalTailWinsWithHonestTotals() throws {
    var run = state()
    for i in 0..<2 {
      run.begin(i)
      try run.receive(
        [
          "type": .string("response.output_text.delta"), "sequence_number": .number(1),
          "delta": .string("Partial"),
        ], elapsed: 0.5)
      try run.receive(Self.terminal("Final page \(i) 🦊"), elapsed: 1)
      try run.finish()
    }
    #expect(run.pages.allSatisfy { $0["state"] == .string("done") })
    #expect(run.text == "## Page 7\n\nFinal page 0 🦊\n\n## Page 8\n\nFinal page 1 🦊")
    #expect(run.usage(seconds: 4)?["promptTokens"]?.integer == 20)
    #expect(run.usage(seconds: 4)?["completionTokens"]?.integer == 10)
    #expect(run.usage(seconds: 4)?["timingSource"]?.string == "end-to-end")
    #expect(run.usage(seconds: 4)?["ttftMs"]?.integer == 500)
    #expect(run.usage(seconds: 4)?["decodeMs"] == nil)
  }
  @Test func missingTerminalCancellationAndRecoveryNeverClaimDone() throws {
    var run = state(3)
    run.begin(0)
    try run.receive(Self.terminal("Saved"), elapsed: 1)
    try run.finish()
    run.begin(1)
    try run.receive(
      ["type": .string("response.output_text.delta"), "delta": .string("Partial")], elapsed: 2)
    #expect(throws: ConversationFailure.self) { try run.finish() }
    try run.flush()
    let message: O = [
      "docRun": run.snapshot, "streaming": .bool(true), "futureField": .string("keep"),
    ]
    let recovered = NativeDocumentRunState.recover(message)
    let pages = try #require(recovered["docRun"]?["pages"]?.array)
    #expect(pages[0]["state"] == .string("done"))
    #expect(pages[1]["state"] == .string("error") && pages[1]["text"] == .string("Partial"))
    #expect(pages[2]["note"] == .string("Interrupted"))
    #expect(recovered["stopped"] == .bool(true) && recovered["futureField"] == .string("keep"))
    #expect(NativeDocumentRunState.recover(recovered) == recovered)
    run.stopRemaining("Stopped")
    #expect(run.pages[0]["state"] == .string("done"))
    #expect(run.pages[1]["text"] == .string("Partial"))
    #expect(run.pages[2]["note"] == .string("Stopped"))
    #expect(run.usage(seconds: 2) == nil)
  }
  @Test func errorsIncompletePagesAndInvalidUsageRemainExplicit() throws {
    var run = state(3)
    run.begin(0)
    try run.receive(Self.terminal("Incomplete", status: "incomplete"), elapsed: 1)
    try run.finish()
    #expect(run.pages[0]["state"] == .string("review"))
    #expect(run.pages[0]["note"] == .string("The model returned an incomplete page"))
    run.begin(1)
    run.fail("Provider unavailable", index: 1)
    run.begin(2)
    try run.receive(Self.terminal("Later page"), elapsed: 2)
    try run.finish()
    #expect(run.pages[1]["state"] == .string("error"))
    #expect(run.pages[2]["state"] == .string("done"))
    #expect(run.usage(seconds: 3) == nil)
    for bad: O in [
      ["input_tokens": .number(-1), "output_tokens": .number(2)],
      ["input_tokens": .number(Decimal(Int.max)), "output_tokens": .number(2)],
      [
        "input_tokens": .number(1), "output_tokens": .number(2),
        "output_tokens_details": .object(["reasoning_tokens": .number(3)]),
      ],
    ] {
      var invalid = state(1)
      invalid.begin(0)
      try invalid.receive(Self.terminal("Done", usage: bad), elapsed: 1)
      try invalid.finish()
      #expect(invalid.usage(seconds: 1) == nil)
    }
  }
  @Test func confidenceUsesTheSameWordFoldingAsWebAndNeverScoresAnUnseenTail() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "ocr-confidence", withExtension: "json", subdirectory: "Fixtures"))
    let cases = try JSONDecoder().decode([V].self, from: Data(contentsOf: url))
    for test in cases {
      let entries = try #require(test["entries"]?.array)
      var confidence = NativeOCRConfidence()
      for entry in entries { confidence.append([entry]) }
      let text = entries.compactMap { $0["token"]?.string }.joined()
      let words = try #require(confidence.words(matching: text))
      let expected = try #require(test["expected"]?.array)
      #expect(words.count == expected.count)
      for (word, want) in zip(words, expected) {
        #expect(word["w"] == want["w"])
        #expect(abs((word["c"]?.double ?? -1) - (want["c"]?.double ?? -2)) < 1e-9)
      }
      #expect(confidence.words(matching: text + " missing tail") == nil)
    }
    var invalid = NativeOCRConfidence()
    invalid.append([.object(["token": .string("bad"), "logprob": .number(1)])])
    #expect(invalid.words(matching: "bad") == nil)
  }
  @Test func regionGeometryIsValidatedBeforeNativeDecoding() {
    let box: V = .array([.number(0), .number(10), .number(999), .number(100)])
    let regions = NativeDocumentRunState.regions([
      .object([
        "label": .string("figure"), "text": .number(7),
        "boxes": .array([
          box, .array([.number(-1), .number(0), .number(10), .number(20)]), .array([]),
        ]),
      ])
    ])
    #expect(regions.first?["boxes"] == .array([box]))
    #expect(regions.first?["text"] == .string(""))
  }
  @Test func closedGroundingMarkersDoNotLeakIntoReadableMarkdown() {
    #expect(
      NativeOCRText.display("<|grounding|><|det|>text [1, 2, 3, 4]<|/det|> **Hello**")
        == "**Hello**")
    #expect(NativeOCRText.display("<|ref|>figure<|/ref|><|det|>[[1,2,3,4]]<|/det|>") == "figure")
    #expect(NativeOCRText.display("Words<|LOC_BEGIN|><|LOC_1|><|LOC_2|><|LOC_END|>") == "Words")
    #expect(NativeOCRText.display("Plain $x$ **text**") == "Plain $x$ **text**")
    #expect(NativeOCRText.display("<|det|>partial") == "<|det|>partial")
  }
}

struct DocumentPlanTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  let parser: O = ["document_parser": .bool(true)]
  let tagged: O = ["task_tags": .array([.object(["tag": .string("<ocr>")])])]
  func user(_ id: String, parent: String? = nil, parts: [V]) -> V {
    .object([
      "id": .string(id), "role": .string("user"), "parentId": parent.map(V.string) ?? .null,
      "content": .array(parts), "createdAt": .number(1),
    ])
  }
  func document(_ messages: [V], leaf: String, selected: String? = nil) throws
    -> ConversationDocument
  {
    try .init(fields: [
      "id": .string("doc"), "model": .string("local"), "messages": .array(messages),
      "leafId": .string(leaf), "activeDocId": selected.map(V.string) ?? .null,
    ])
  }
  let pdf: V = .object([
    "type": .string("file"), "name": .string("Book.pdf"), "attachmentId": .string("pdf"),
  ])
  let image: V = .object(["type": .string("image"), "attachmentId": .string("image")])
  func text(_ value: String) -> V { .object(["type": .string("text"), "text": .string(value)]) }
  @Test func selectionIsStickyBranchLocalAndNewDocumentsWin() throws {
    let a = user("a", parts: [pdf])
    let b = user("b", parent: "a", parts: [image])
    let c = user("c", parent: "b", parts: [text("Read again")])
    let sibling = user("sibling", parts: [image])
    #expect(
      NativeDocumentPlan.make(
        document: try document([a, b, c, sibling], leaf: "c", selected: "a"), capability: parser)?
        .sourceID == "a")
    #expect(
      NativeDocumentPlan.make(
        document: try document([a, b, c, sibling], leaf: "c", selected: "sibling"),
        capability: parser)?.sourceID == "b")
    #expect(
      NativeDocumentPlan.make(
        document: try document([a, b], leaf: "b", selected: "a"), capability: parser)?.sourceID
        == "b")
    let docx = user(
      "word", parent: "b",
      parts: [.object(["type": .string("file"), "name": .string("Letter.docx")])])
    #expect(
      NativeDocumentPlan.make(
        document: try document([a, b, docx], leaf: "word", selected: "a"), capability: parser)
        == nil)
  }
  @Test func ordinaryVisionAndSingleImageTaskStayOnTheConversationPath() throws {
    let doc = try document([user("a", parts: [pdf, text("<ocr>")])], leaf: "a")
    #expect(NativeDocumentPlan.make(document: doc, capability: ["vision": .bool(true)]) == nil)
    #expect(NativeDocumentPlan.make(document: doc, capability: tagged)?.parts.count == 1)
    let single = try document([user("a", parts: [image, text("<ocr>")])], leaf: "a")
    #expect(NativeDocumentPlan.make(document: single, capability: tagged) == nil)
    let followup = try document(
      doc.messages.map(V.object) + [user("b", parent: "a", parts: [text("Explain this")])],
      leaf: "b")
    #expect(NativeDocumentPlan.make(document: followup, capability: tagged) == nil)
    #expect(NativeDocumentPlan.make(document: single, capability: parser)?.parts.count == 1)
  }
  @Test func explicitInclusivePageRangesNeverTruncateSilently() throws {
    #expect(try NativeDocumentPlan.pages("7-9", count: 200, limit: 40) == 7...9)
    #expect(try NativeDocumentPlan.pages("199-", count: 200, limit: 40) == 199...200)
    #expect(try NativeDocumentPlan.pages("8", count: 200, limit: 40) == 8...8)
    for value in ["0-3", "4-3", "1-201", "1-41", "broken", "1-2-3", ""] {
      #expect(throws: ConversationFailure.self) {
        try NativeDocumentPlan.pages(value, count: 200, limit: 40)
      }
    }
  }
}
