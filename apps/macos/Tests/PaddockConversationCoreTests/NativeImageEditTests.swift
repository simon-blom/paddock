import Foundation
import Testing

@testable import PaddockConversationCore

@Suite("Native image-edit reference policy")
struct NativeImageEditTests {
  typealias V = ConversationValue
  typealias O = [String: V]
  let caps: O = ["edit": .bool(true), "max_references": .number(2)]
  func picture(_ id: String, generated: Bool = false, preview: Bool = false) -> V {
    var p: O = [
      "type": .string("image"), "attachmentId": .string(id), "mime": .string("image/png"),
    ]
    if generated { p["gen"] = .object(["seed": .number(42), "preview": .bool(preview)]) }
    return .object(p)
  }
  func message(
    _ id: String, _ parent: String?, _ role: String, _ text: String = "", images: [V] = []
  ) -> O {
    var m: O = [
      "id": .string(id), "parentId": parent.map(V.string) ?? .null,
      "role": .string(role),
      "content": .array(
        (text.isEmpty ? [] : [.object(["type": .string("text"), "text": .string(text)])]) + images),
    ]
    if role == "assistant", !images.isEmpty { m["imageGen"] = .object(["seed": .number(42)]) }
    return m
  }
  func doc(_ messages: [O], leaf: String = "answer") throws -> ConversationDocument {
    try ConversationDocument(fields: [
      "id": .string("chat"), "leafId": .string(leaf), "messages": .array(messages.map(V.object)),
    ])
  }
  @Test func attachmentsWinInOrderAndOnlyLatestInstructionIsSent() throws {
    let answer = message("answer", "edit", "assistant")
    let d = try doc([
      message("user", nil, "user", "An apple"),
      message("old", "user", "assistant", images: [picture("previous", generated: true)]),
      message(
        "edit", "old", "user", "  Make it blue  ", images: [picture("first"), picture("second")]),
      answer,
    ])
    let plan = try NativeImageGeneration.plan(document: d, message: answer, caps: caps)
    #expect(plan.prompt == "Make it blue" && plan.from == "attached")
    #expect(plan.references.map { $0["attachmentId"] } == [.string("first"), .string("second")])
  }
  @Test func followupUsesLastCompletedPictureNotPreviewOrAnotherBranch() throws {
    let answer = message("answer", "edit", "assistant")
    let d = try doc([
      message("user", nil, "user", "An apple"),
      message(
        "old", "user", "assistant",
        images: [
          picture("first", generated: true), picture("last", generated: true),
          picture("preview", generated: true, preview: true),
        ]),
      message("abandoned", "user", "assistant", images: [picture("wrong", generated: true)]),
      message("edit", "old", "user", "Make it blue"), answer,
    ])
    let plan = try NativeImageGeneration.plan(document: d, message: answer, caps: caps)
    #expect(plan.prompt == "Make it blue" && plan.from == "previous")
    #expect(plan.references.map { $0["attachmentId"] } == [.string("last")])
    let textOnly = try NativeImageGeneration.plan(document: d, message: answer, caps: [:])
    #expect(textOnly.references.isEmpty && textOnly.prompt == "An apple\nMake it blue")
  }
  @Test func compareCompletionOrderCannotChangeTheOtherLanesReference() throws {
    var one = message("one", "edit", "assistant", images: [picture("too-new", generated: true)])
    var two = message("answer", "edit", "assistant")
    one["group"] = .string("comparison")
    two["group"] = .string("comparison")
    let d = try doc(
      [
        message("user", nil, "user", "An apple"),
        message("old", "user", "assistant", images: [picture("previous", generated: true)]),
        message("edit", "old", "user", "Blue"), one, two,
      ], leaf: "one")
    let a = try NativeImageGeneration.plan(document: d, message: one, caps: caps)
    let b = try NativeImageGeneration.plan(document: d, message: two, caps: caps)
    #expect(
      a.references == b.references && b.references.first?["attachmentId"] == .string("previous"))
  }
  @Test func retryUsesItsParentReferenceNotTheResultBeingRetried() throws {
    let answer = message("answer", "edit", "assistant")
    let d = try doc([
      message("user", nil, "user", "An apple"),
      message("old", "user", "assistant", images: [picture("original", generated: true)]),
      message("edit", "old", "user", "Blue"),
      message(
        "retry-source", "edit", "assistant", images: [picture("do-not-edit", generated: true)]),
      answer,
    ])
    #expect(
      try NativeImageGeneration.plan(document: d, message: answer, caps: caps).references.first?[
        "attachmentId"] == .string("original"))
  }
  @Test func unsafeOrUnsupportedReferencesAreExplicitlyRejected() throws {
    let image = picture("original").object!
    for (parts, cap) in [
      ([image], O()), ([image, image, image], caps),
      ([["type": .string("file"), "mime": .string("application/pdf")]], caps),
      ([image.merging(["unreadable": .bool(true)]) { _, new in new }], caps),
    ] {
      #expect(throws: (any Error).self) {
        try NativeImageGeneration.validateReferences(parts, caps: cap)
      }
    }
    let answer = message("answer", "user", "assistant")
    let d = try doc([message("user", nil, "user", images: [picture("original")]), answer])
    #expect(throws: (any Error).self) {
      try NativeImageGeneration.plan(document: d, message: answer, caps: caps)
    }
  }
}

@Suite("Bounded native image multipart")
struct NativeImageMultipartTests {
  @Test func scalarFieldsAndBinaryOriginalsUseTheWebFormWithoutBase64Expansion() throws {
    let form = try NativeImageMultipart(fields: [
      "prompt": .string("Blå 🦊\nbackground"), "seed": .number(42), "stream": .bool(true),
    ])
    let directory = form.directory
    defer { form.discard() }
    // A non-zero-index Data slice must preserve its complete byte sequence.
    let first = Data([9, 0, 255, 13, 10]).dropFirst()
    try form.add(first, mime: "image/png")
    try form.add(Data([1, 2, 3]), mime: "image/jpeg")
    let data = try form.finish()
    #expect(data.range(of: first) != nil)
    let text = String(decoding: data, as: UTF8.self)
    #expect(text.contains("name=\"prompt\"\r\n\r\nBlå 🦊\nbackground\r\n"))
    #expect(text.contains("name=\"stream\"\r\n\r\ntrue\r\n"))
    #expect(text.contains("name=\"seed\"\r\n\r\n42\r\n"))
    #expect(text.components(separatedBy: "name=\"image[]\"").count == 3)
    #expect(text.hasSuffix("--\(form.boundary)--\r\n"))
    let attributes = try FileManager.default.attributesOfItem(atPath: directory.path)
    #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o700)
    form.discard()
    #expect(!FileManager.default.fileExists(atPath: directory.path))
  }
  @Test func limitsAndHeaderInjectionAreCheckedBeforeAppendingBytes() throws {
    let form = try NativeImageMultipart(fields: ["prompt": .string("Edit")], maximumBytes: 4096)
    defer { form.discard() }
    let before = form.count
    #expect(throws: (any Error).self) {
      try form.add(Data(repeating: 0, count: 4096), mime: "image/png")
    }
    #expect(throws: (any Error).self) {
      try form.add(Data([1]), mime: "image/png\r\nX-Injected: yes")
    }
    #expect(throws: (any Error).self) { try form.add(Data(), mime: "image/png") }
    #expect(form.count == before)
    #expect(throws: (any Error).self) { try form.finish() }
    #expect(throws: (any Error).self) {
      try NativeImageMultipart(fields: ["prompt\r\n": .string("bad")])
    }
  }
  @Test func droppingTheOwnerRemovesAnUnfinishedUpload() throws {
    var form: NativeImageMultipart? = try NativeImageMultipart(fields: [:])
    let directory = form!.directory
    try form!.add(Data([1]), mime: "image/png")
    form = nil
    #expect(!FileManager.default.fileExists(atPath: directory.path))
  }
  @Test func cancellationStopsWritingAndReleasesThePrivateSpool() async throws {
    let directory = try await Task {
      let form = try NativeImageMultipart(fields: ["prompt": .string("Blue")])
      defer { form.discard() }
      withUnsafeCurrentTask { $0?.cancel() }
      #expect(throws: CancellationError.self) {
        try form.add(Data(repeating: 1, count: 1024), mime: "image/png")
      }
      return form.directory
    }.value
    #expect(!FileManager.default.fileExists(atPath: directory.path))
  }
}
