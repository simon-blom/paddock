import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Native image generation wire contract")
struct NativeImageGenerationTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  let caps: O = [
    "max_n": .number(4), "max_steps": .number(100),
    "output_formats": .array([.string("png"), .string("jpeg")]),
    "stream": .bool(true), "max_partial_images": .number(3),
  ]
  @Test func retryNeverSilentlyDiscardsReferenceImages() throws {
    func user(_ text: String) -> O {
      [
        "role": .string("user"),
        "content": .array([
          .object(["type": .string("text"), "text": .string(text)])
        ]),
      ]
    }
    let first = user("A red apple")
    let followup = user("on a white table")
    #expect(
      try NativeImageGeneration.textPrompt([first, followup]) == "A red apple\non a white table")
    var attached = followup
    attached["content"] = .array(
      (attached["content"]?.array ?? []) + [
        .object(["type": .string("image"), "attachmentId": .string("original")])
      ])
    #expect(throws: (any Error).self) {
      try NativeImageGeneration.textPrompt([first, attached])
    }
    let generated: O = [
      "role": .string("assistant"),
      "content": .array([
        .object(["type": .string("image"), "gen": .object(["seed": .number(42)])])
      ]),
    ]
    #expect(
      try NativeImageGeneration.textPrompt([first, generated, followup])
        == "A red apple\non a white table")
  }
  @Test func batchTimingMatchesWebPerPicturePerStep() {
    #expect(NativeImageGeneration.secondsPerStep(elapsed: 80, steps: 40, images: 2) == 1)
    #expect(NativeImageGeneration.secondsPerStep(elapsed: 40, steps: 40, images: 1) == 1)
  }
  @Test func defaultsAndBatchingMatchWebStudio() throws {
    var params = NativeImageGeneration.defaults
    let body = try NativeImageGeneration.body(
      model: "image", prompt: "Apple", params: params, seed: 42, caps: caps)
    #expect(body["stream"] == .bool(true) && body["partial_images"] == .number(2))
    #expect(body["size"] == nil && body["steps"] == nil && body["quality"] == nil)
    #expect(body["seed"] == .number(42) && body["max_output_tokens"] == nil)
    params["n"] = .number(4)
    let batch = try NativeImageGeneration.body(
      model: "image", prompt: "Apple", params: params, seed: 42, caps: caps)
    #expect(batch["n"] == .number(4) && batch["stream"] == nil)
  }
  @Test func invalidSettingsAndEndpointLimitsAreRejected() throws {
    for patch: O in [
      ["size": .string("1025x1024")], ["size": .string("32x32x")], ["steps": .number(0)],
      ["seed": .number(-1)], ["steps": .number(2.5)], ["n": .number(5)],
      ["format": .string("jpeg"), "background": .string("transparent")],
    ] {
      let params = NativeImageGeneration.defaults.merging(patch) { _, new in new }
      #expect(throws: (any Error).self) { try NativeImageGeneration.validate(params) }
    }
    var params = NativeImageGeneration.defaults
    params["n"] = .number(2)
    #expect(throws: (any Error).self) {
      try NativeImageGeneration.body(
        model: "i", prompt: "Apple", params: params, seed: 1, caps: [:])
    }
  }
  @Test func threadRetryAndCompareSeedsMatchTheWebPolicy() throws {
    let user: O = ["id": .string("u"), "role": .string("user"), "content": .array([])]
    let old: O = [
      "id": .string("a"), "role": .string("assistant"), "parentId": .string("u"),
      "content": .array([]), "imageGen": .object(["seed": .number(42)]),
    ]
    let nextUser: O = [
      "id": .string("u2"), "role": .string("user"), "parentId": .string("a"), "content": .array([]),
    ]
    var next: O = [
      "id": .string("a2"), "role": .string("assistant"), "parentId": .string("u2"),
      "content": .array([]),
    ]
    var doc = try ConversationDocument(fields: [
      "id": .string("c"), "leafId": .string("a2"),
      "messages": .array([user, old, nextUser, next].map(V.object)),
    ])
    #expect(
      NativeImageGeneration.seed(NativeImageGeneration.defaults, document: doc, message: next) {
        99
      } == 42)
    #expect(
      NativeImageGeneration.seed(
        NativeImageGeneration.defaults, document: doc, message: next, editing: true
      ) { 99 } == 99)
    var pinned = NativeImageGeneration.defaults
    pinned["seed"] = .number(42)
    #expect(
      NativeImageGeneration.seed(pinned, document: doc, message: next, editing: true) { 99 }
        == 42)
    pinned["seed"] = .number(0)
    #expect(
      NativeImageGeneration.seed(pinned, document: doc, message: next, editing: true) { 99 }
        == 0)
    next["parentId"] = .string("u")
    #expect(
      NativeImageGeneration.seed(NativeImageGeneration.defaults, document: doc, message: next) {
        99
      } == 99)
    next["group"] = .string("comparison")
    doc = try ConversationDocument(fields: [
      "id": .string("c"), "messages": .array([user, next].map(V.object)),
    ])
    let one = NativeImageGeneration.seed(
      NativeImageGeneration.defaults, document: doc, message: next
    ) { 99 }
    #expect(
      one
        == NativeImageGeneration.seed(NativeImageGeneration.defaults, document: doc, message: next)
      { 20 })
    #expect(one == 754_347_588)
    next["group"] = .string("compare-🌱")
    #expect(
      NativeImageGeneration.seed(
        NativeImageGeneration.defaults, document: doc, message: next, editing: true
      ) { 20 } == 193_383_420)
  }
}
