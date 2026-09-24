import Foundation
import Testing

@testable import PaddockConversationCore

struct ContextPlanTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  static func document(_ row: O = [:]) throws -> ConversationDocument {
    var messages: [V] = []
    for (i, count) in [6000, 6000, 1600, 1600].enumerated() {
      messages.append(
        .object([
          "id": .string("m\(i)"), "parentId": i == 0 ? .null : .string("m\(i-1)"),
          "role": .string(i % 2 == 0 ? "user" : "assistant"), "model": .string("cloud:ep:remote"),
          "content": .array([
            .object([
              "type": .string("text"), "text": .string(String(repeating: "a", count: count)),
            ])
          ]),
        ]))
    }
    var fields: O = [
      "id": .string("context-fixture"), "model": .string("cloud:ep:remote"),
      "messages": .array(messages), "leafId": .string("m3"),
      "params": .object(NativeStudioRuntime.defaultParams),
    ]
    if row["saved"]?.bool == true || row["stale"]?.bool == true {
      fields.merge([
        "summary": .string("Brief"), "summaryCount": .number(2),
        "summaryLastId": .string(row["stale"]?.bool == true ? "other" : "m1"),
      ]) { _, new in new }
    }
    if row["item"]?.bool == true {
      fields["serverCompaction"] = .object([
        "id": .string("sc"), "content": .string("Brief"), "tailStartId": .string("m2"),
      ])
    }
    return try ConversationDocument(fields: fields)
  }
  @Test func sharesWebContextPlanningFixtures() throws {
    let url = try #require(
      Bundle.module.url(
        forResource: "context-plans", withExtension: "json", subdirectory: "Fixtures"))
    for row in try JSONDecoder().decode([O].self, from: Data(contentsOf: url)) {
      let doc = try Self.document(row)
      let ctx = row["context"]!.integer!
      let reply = row["reply"]!.integer!
      let plan = NativeContextPlan.resolve(
        doc, context: ctx, reply: reply,
        summarize: row["summarize"]!.bool!, server: row["server"]!.bool!)
      #expect(plan.from == row["from"]!.integer!, "\(row)")
      #expect(plan.threshold == row["threshold"]!.integer!)
      #expect(plan.summary == row["summary"]?.string)
      #expect(NativeContextPlan.contextTokens(doc) == row["used"]!.integer!)
      #expect(
        NativeContextPlan.compactionTarget(doc, context: ctx, reply: reply) == row["target"]!
          .integer!)
    }
  }
  @Test func neverDropsTheNewestUserForAnEmptyPlaceholder() throws {
    var fields = try Self.document().fields
    var rows = fields["messages"]!.array!
    var question = rows[2].object!
    question["content"] = .array([
      .object(["type": .string("text"), "text": .string(String(repeating: "x", count: 36000))])
    ])
    rows[2] = .object(question)
    var pending = rows[3].object!
    pending["content"] = .array([])
    rows[3] = .object(pending)
    fields["messages"] = .array(rows)
    let plan = NativeContextPlan.resolve(
      try .init(fields: fields), context: 8192, reply: 4096, summarize: true, server: false)
    #expect(plan.from == 2)
  }
  @Test func modelProvenanceAndBranchInvalidateSummary() throws {
    var fields = try Self.document(["saved": .bool(true)]).fields
    fields["summaryModel"] = .string("another-model")
    #expect(!NativeContextPlan.summaryValid(try .init(fields: fields)))
    fields["summaryModel"] = fields["model"]
    #expect(NativeContextPlan.summaryValid(try .init(fields: fields)))
    fields["leafId"] = .string("m0")
    #expect(!NativeContextPlan.summaryValid(try .init(fields: fields)))
  }
  @Test func contextMeterAnchorsReportedUsageButNotAccumulatedToolOrDocumentBills() throws {
    var fields = try Self.document().fields
    var messages = fields["messages"]!.array!
    var last = messages[3].object!
    last["usage"] = .object(["promptTokens": .number(7000), "completionTokens": .number(100)])
    messages[3] = .object(last)
    fields["messages"] = .array(messages)
    #expect(NativeContextPlan.contextTokens(try .init(fields: fields)) == 7104)
    #expect(NativeContextPlan.contextTokens(try .init(fields: fields), draft: "Next") == 7109)
    last["docRun"] = .object(["pages": .array([])])
    messages[3] = .object(last)
    fields["messages"] = .array(messages)
    #expect(NativeContextPlan.contextTokens(try .init(fields: fields)) == 3816)
    last["docRun"] = nil
    last["toolCalls"] = .array([.object(["name": .string("search")])])
    messages[3] = .object(last)
    fields["messages"] = .array(messages)
    #expect(NativeContextPlan.contextTokens(try .init(fields: fields)) == 3816)
  }
  @Test func exactBudgetRequiresSameWirePrefixAndNoAggregatedUsage() throws {
    let input: [V] = [
      .object([
        "type": .string("message"), "role": .string("user"),
        "content": .array([.object(["type": .string("input_text"), "text": .string("Hello")])]),
      ])
    ]
    var body: O = [
      "model": .string("cloud"), "input": .array(input), "instructions": .string("System"),
      "tools": .array([]),
    ]
    let key = try #require(NativeContextPlan.prefixKey(body, count: 1))
    var fields = try Self.document().fields
    fields["messages"] = .array([
      .object([
        "id": .string("a"), "parentId": .null, "role": .string("assistant"),
        "content": .array([]),
        "usage": .object(["promptTokens": .number(7000), "completionTokens": .number(0)]),
        "run": .object([
          "model": .string("cloud"),
          "nativeContext": .object(["count": .number(1), "key": .string(key)]),
        ]),
      ])
    ])
    fields["leafId"] = .string("a")
    var doc = try ConversationDocument(fields: fields)
    #expect(
      NativeContextPlan.replyPrompt(doc, body: body, model: "cloud", pending: "new").exact == 7000)
    body["instructions"] = .string("Changed")
    #expect(
      NativeContextPlan.replyPrompt(doc, body: body, model: "cloud", pending: "new").exact == 0)
    body["instructions"] = .string("System")
    var row = fields["messages"]!.array![0].object!
    row["toolCalls"] = .array([.object(["name": .string("search")])])
    fields["messages"] = .array([.object(row)])
    doc = try .init(fields: fields)
    #expect(
      NativeContextPlan.replyPrompt(doc, body: body, model: "cloud", pending: "new").exact == 0)
    #expect(
      NativeContextPlan.prefixKey(
        [
          "input": .array([
            .object([
              "type": .string("message"),
              "content": .array([.object(["type": .string("input_image")])]),
            ])
          ])
        ], count: 1) == nil)
  }
  @Test func summaryRequestIsBoundedAndExcludesPrivatePayloads() throws {
    let doc = try Self.document()
    #expect(
      NativeStudioRuntime.summaryInput(doc, count: 2, context: 2048, model: "cloud:ep:remote")
        .isEmpty)
    let text = NativeStudioRuntime.summaryInput(
      doc, count: 2, context: 3000, model: "cloud:ep:remote")
    #expect(text.count <= (3000 - 2688) * 4)
    #expect(!text.contains("data:image"))
  }
  @Test func correctionOnlyAcceptsNumericPreflightOverflow() throws {
    func error(_ code: Int, _ message: String) throws -> ConversationFailure {
      .invalid(
        String(
          decoding: try JSONEncoder().encode([
            "code": V.number(Decimal(code)), "message": .string(message),
          ]), as: UTF8.self))
    }
    let refused = try error(
      400, "input length and max_tokens exceed context limit: 7000 + 4000 > 8192")
    #expect(NativeContextPlan.correctedLimit(error: refused, previous: 4000, ceiling: nil) == 1192)
    #expect(NativeContextPlan.correctedLimit(error: refused, previous: 1000, ceiling: nil) == nil)
    #expect(
      try NativeContextPlan.correctedLimit(
        error: error(429, "exceed context limit: 7000 + 4000 > 8192"), previous: 4000, ceiling: nil)
        == nil)
    #expect(
      try NativeContextPlan.correctedLimit(
        error: error(400, "exceed context limit: 8100 + 4000 > 8192"), previous: 4000, ceiling: nil)
        == nil)
  }
}
