import Foundation
import Testing

@testable import PaddockConversationCore

extension RuntimeTests {
  var editingCaps: O {
    [
      "image_generation": .object([
        "edit": .bool(true), "max_references": .number(2),
        "stream": .bool(true), "max_n": .number(4), "max_steps": .number(100),
        "max_partial_images": .number(3), "default_steps": .number(40),
        "default_size": .string("1024x1024"), "output_formats": .array([.string("png")]),
      ])
    ]
  }
  @Test func imageEditUsesOriginalsAndFollowupAfterReopenUsesGeneratedPicture() async throws {
    let runtime = try await fixture(localCapabilities: editingCaps)
    _ = try await runtime.command("models", ["ids": .array([.string("local")])])
    #expect(await runtime.presentation()["composer"]?["imageEditing"] == .bool(true))
    RuntimeProtocol.state.withLock {
      $0.imageAttachments["reference-one"] = Data("first-original".utf8)
      $0.imageAttachments["reference-two"] = Data("second-original".utf8)
    }
    for id in ["reference-one", "reference-two"] {
      _ = try await runtime.command(
        "stage",
        [
          "id": .string(id), "mime": .string("image/png"), "name": .string("\(id).png"),
          "size": .number(20),
        ])
    }
    #expect(await runtime.presentation()["composer"]?["inputIssue"] == .string(""))
    _ = try await runtime.command(
      "send",
      [
        "text": .string("Combine these pictures"),
        "attachments": .array([
          .object(["id": .string("reference-one")]), .object(["id": .string("reference-two")]),
        ]),
      ])
    _ = try await settled(runtime)
    let fields = try #require(await runtime.currentFields())
    let answer = try #require(fields["messages"]?.array?.last)
    #expect(answer["error"] == nil)
    #expect(answer["imageGen"]?["references"] == .number(2))
    #expect(answer["imageGen"]?["referencesFrom"] == .string("attached"))
    #expect(answer["imageGen"]?["previews"] == .number(1))
    let first = try #require(
      RuntimeProtocol.state.withLock {
        $0.requests.first { $0["path"]?.string?.hasSuffix("/images/edits") == true }
      })
    let form = try #require(first["multipart"]?.string)
    #expect(first["contentType"]?.string?.hasPrefix("multipart/form-data; boundary=") == true)
    #expect(form.contains("first-original") && form.contains("second-original"))
    #expect(
      form.range(of: "first-original")!.lowerBound < form.range(of: "second-original")!.lowerBound)
    #expect(
      !form.contains("name=\"size\""),
      "Auto edit size must be chosen from the reference by the endpoint")
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": fields["id"]!])
    _ = try await runtime.command("send", ["text": .string("Make the sky blue")])
    _ = try await settled(runtime)
    let followup = try #require(await runtime.currentFields()?["messages"]?.array?.last)
    #expect(followup["error"] == nil)
    #expect(followup["imageGen"]?["prompt"] == .string("Make the sky blue"))
    #expect(followup["imageGen"]?["referencesFrom"] == .string("previous"))
    #expect(followup["imageGen"]?["references"] == .number(1))
    #expect(followup["imageGen"]?["seed"] != answer["imageGen"]?["seed"])
    let requests = RuntimeProtocol.state.withLock { $0.requests }
    #expect(requests.filter { $0["path"]?.string?.hasSuffix("/images/edits") == true }.count == 2)
    let pictureID = try #require(answer["content"]?.array?.first?["attachmentId"]?.string)
    #expect(
      requests.contains {
        $0["method"] == .string("GET") && $0["path"] == .string("/api/attachments/\(pictureID)")
      })
    #expect(!requests.contains { $0["path"]?.string?.hasSuffix("/images/generations") == true })
    await runtime.close()
  }
  @Test func imageEditingPreservesInlineOriginalAfterStorageFailure() async throws {
    let runtime = try await fixture(localCapabilities: editingCaps)
    RuntimeProtocol.state.withLock { $0.failImageStore = true }
    _ = try await runtime.command("models", ["ids": .array([.string("local")])])
    _ = try await runtime.command("send", ["text": .string("An apple")])
    _ = try await settled(runtime)
    let initial = try #require(await runtime.currentFields())
    let picture = try #require(initial["messages"]?.array?.last?["content"]?.array?.first)
    #expect(picture["attachmentId"] == .string(""))
    _ = try await runtime.command("newChat")
    _ = try await runtime.command("open", ["id": initial["id"]!])
    _ = try await runtime.command("send", ["text": .string("Make it blue")])
    _ = try await settled(runtime)
    let next = try #require(await runtime.currentFields()?["messages"]?.array?.last)
    #expect(next["error"] == nil && next["imageGen"]?["referencesFrom"] == .string("previous"))
    #expect(
      next["content"]?.array?.first?["dataUrl"]?.string?.hasPrefix("data:image/png;base64,") == true
    )
    await runtime.close()
  }
  @Test func multipleEditedPicturesUseJSONThenFollowupEditsTheLastResult() async throws {
    let runtime = try await fixture(localCapabilities: editingCaps)
    RuntimeProtocol.state.withLock {
      $0.imageJSONCount = 2
      $0.imageAttachments["original"] = Data("original".utf8)
    }
    _ = try await runtime.command("models", ["ids": .array([.string("local")])])
    _ = try await runtime.command("settings", ["imageParams": .object(["n": .number(2)])])
    _ = try await runtime.command(
      "stage",
      [
        "id": .string("original"), "mime": .string("image/png"), "name": .string("Reference.png"),
        "size": .number(8),
      ])
    _ = try await runtime.command(
      "send",
      [
        "text": .string("Blue"), "attachments": .array([.object(["id": .string("original")])]),
      ])
    _ = try await settled(runtime)
    let answer = try #require(await runtime.currentFields()?["messages"]?.array?.last)
    #expect(answer["error"] == nil && answer["content"]?.array?.count == 2)
    #expect(answer["imageGen"]?["size"] == .string("1536x1024"))
    #expect(answer["imageGen"]?["previews"] == .number(0))
    let form = try #require(
      RuntimeProtocol.state.withLock {
        $0.requests.first { $0["path"]?.string?.hasSuffix("/images/edits") == true }?["multipart"]?
          .string
      })
    #expect(form.contains("name=\"n\"\r\n\r\n2\r\n"))
    #expect(!form.contains("name=\"stream\"") && !form.contains("name=\"partial_images\""))
    RuntimeProtocol.state.withLock { $0.requests = [] }
    _ = try await runtime.command("send", ["text": .string("Green")])
    _ = try await settled(runtime)
    let pictures = try #require(answer["content"]?.array)
    let fetched = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["method"] == .string("GET") }.compactMap { $0["path"]?.string }
    }
    #expect(fetched.contains("/api/attachments/\(pictures[1]["attachmentId"]!.string!)"))
    #expect(!fetched.contains("/api/attachments/\(pictures[0]["attachmentId"]!.string!)"))
    #expect(await runtime.currentFields()?["messages"]?.array?.last?["error"] == nil)
    await runtime.close()
  }
  @Test func compareRequiresBothEditingCapabilitiesAndSharesReferencesAndSeed() async throws {
    for both in [false, true] {
      var peer = editingCaps
      var image = peer["image_generation"]!.object!
      image["edit"] = .bool(both)
      peer["image_generation"] = .object(image)
      let runtime = try await fixture(localCapabilities: editingCaps, imagePeerCapabilities: peer)
      _ = try await runtime.command(
        "models", ["ids": .array([.string("local"), .string("local-peer")])])
      #expect(await runtime.presentation()["composer"]?["imageEditing"] == .bool(both))
      RuntimeProtocol.state.withLock {
        $0.imageAttachments["original"] = Data("shared-original".utf8)
      }
      _ = try await runtime.command(
        "stage",
        [
          "id": .string("original"), "mime": .string("image/png"), "name": .string("Reference.png"),
          "size": .number(15),
        ])
      let input: O = [
        "text": .string("Blue"), "attachments": .array([.object(["id": .string("original")])]),
      ]
      if !both {
        await #expect(throws: ConversationFailure.self) { try await runtime.command("send", input) }
        #expect(await runtime.staged["original"] != nil)
      } else {
        _ = try await runtime.command("send", input)
        _ = try await settled(runtime)
        let messages = try #require(await runtime.currentFields()?["messages"]?.array)
        let answers = messages.filter { $0["role"] == .string("assistant") }
        #expect(answers.count == 2 && answers.allSatisfy { $0["error"] == nil })
        #expect(answers[0]["imageGen"]?["seed"] == answers[1]["imageGen"]?["seed"])
        let edits = RuntimeProtocol.state.withLock {
          $0.requests.filter { $0["path"]?.string?.hasSuffix("/images/edits") == true }
        }
        #expect(Set(edits.compactMap { $0["path"]?.string }).count == 2)
        #expect(edits.allSatisfy { $0["multipart"]?.string?.contains("shared-original") == true })
      }
      await runtime.close()
    }
  }
  @Test func editingCapabilityAndCountGuardsPreserveTheDraftBeforeSaving() async throws {
    for edits in [false, true] {
      var caps = editingCaps
      var image = caps["image_generation"]!.object!
      image["edit"] = .bool(edits)
      caps["image_generation"] = .object(image)
      let runtime = try await fixture(localCapabilities: caps)
      _ = try await runtime.command("models", ["ids": .array([.string("local")])])
      #expect(await runtime.presentation()["composer"]?["imageEditing"] == .bool(edits))
      let ids = edits ? ["original", "second", "third"] : ["original"]
      for id in ids {
        _ = try await runtime.command(
          "stage",
          [
            "id": .string(id), "mime": .string("image/png"), "name": .string("\(id).png"),
            "size": .number(12),
          ])
      }
      let before = await runtime.currentFields()
      #expect(await runtime.presentation()["composer"]?["inputIssue"]?.string?.isEmpty == false)
      await #expect(throws: ConversationFailure.self) {
        try await runtime.command(
          "send",
          [
            "text": .string("Blue"),
            "attachments": .array(ids.map { .object(["id": .string($0)]) }),
          ])
      }
      #expect(await runtime.currentFields() == before)
      #expect(await runtime.staged.count == ids.count)
      #expect(
        !RuntimeProtocol.state.withLock {
          $0.requests.contains { $0["path"]?.string?.contains("/images/") == true }
        })
      await runtime.close()
    }
  }
  @Test func retryReusesTheOriginalReferenceAndLatestInstruction() async throws {
    let runtime = try await fixture(localCapabilities: editingCaps)
    _ = try await runtime.command("models", ["ids": .array([.string("local")])])
    _ = try await runtime.command("settings", ["imageParams": .object(["seed": .number(42)])])
    _ = try await runtime.command("send", ["text": .string("An apple")])
    _ = try await settled(runtime)
    let original = try #require(await runtime.currentFields()?["messages"]?.array?.last)
    let reference = try #require(original["content"]?.array?.first?["attachmentId"]?.string)
    _ = try await runtime.command("send", ["text": .string("Blue")])
    _ = try await settled(runtime)
    let doc = try #require(await runtime.currentFields())
    let edited = try #require(
      doc["messages"]?.array?.last?["content"]?.array?.first?["attachmentId"]?.string)
    RuntimeProtocol.state.withLock { $0.requests = [] }
    _ = try await runtime.command(
      "messageAction",
      [
        "action": .string("retry"), "conversationId": doc["id"]!,
        "leafId": doc["leafId"]!, "messageId": doc["leafId"]!,
      ])
    _ = try await settled(runtime)
    let retried = try #require(await runtime.currentFields()?["messages"]?.array?.last)
    #expect(retried["error"] == nil)
    #expect(retried["imageGen"]?["prompt"] == .string("Blue"))
    #expect(retried["imageGen"]?["seed"] == .number(42))
    let fetched = RuntimeProtocol.state.withLock {
      $0.requests.filter { $0["method"] == .string("GET") }.compactMap { $0["path"]?.string }
    }
    #expect(fetched.contains("/api/attachments/\(reference)"))
    #expect(!fetched.contains("/api/attachments/\(edited)"))
    await runtime.close()
  }
  @Test func editTransportRejectsExternalSourcesTraversalAndBadInlineBytes() async throws {
    let runtime = try await fixture(localCapabilities: editingCaps)
    let transport = await runtime.transport
    for part: O in [
      ["modelUrl": .string("https://example.com/private.png")],
      ["attachmentId": .string("../secret")],
      ["dataUrl": .string("data:image/png;base64,%%%")],
      ["dataUrl": .string("data:image/png,not-base64")],
      ["thumbnail": .string("data:image/png;base64,AQ==")],
    ] {
      await #expect(throws: ConversationFailure.self) {
        try await transport.images(
          port: 12481, body: ["prompt": .string("Blue")], references: [part]
        ) { _ in }
      }
    }
    #expect(
      !RuntimeProtocol.state.withLock {
        $0.requests.contains {
          $0["path"]?.string?.contains("/images/") == true
            || $0["path"]?.string?.hasPrefix("/api/attachments/") == true
        }
      })
    await runtime.close()
  }
  @Test func missingOriginalAndEndpointFailureNeverFallBackToTextGeneration() async throws {
    for missing in [true, false] {
      let runtime = try await fixture(localCapabilities: editingCaps)
      RuntimeProtocol.state.withLock {
        if !missing { $0.imageAttachments["original"] = Data("original".utf8) }
        $0.imageEditError = 422
      }
      _ = try await runtime.command("models", ["ids": .array([.string("local")])])
      _ = try await runtime.command(
        "stage",
        [
          "id": .string("original"), "mime": .string("image/png"), "name": .string("Original.png"),
          "size": .number(12),
        ])
      _ = try await runtime.command(
        "send",
        ["text": .string("Blue"), "attachments": .array([.object(["id": .string("original")])])])
      _ = try await settled(runtime)
      let answer = try #require(await runtime.currentFields()?["messages"]?.array?.last)
      #expect(answer["error"]?.string?.isEmpty == false)
      #expect(answer["content"]?.array?.isEmpty == true)
      if !missing { #expect(answer["error"]?.string?.contains("Reference image rejected") == true) }
      #expect(
        !RuntimeProtocol.state.withLock {
          $0.requests.contains { $0["path"]?.string?.hasSuffix("/images/generations") == true }
        })
      await runtime.close()
    }
  }
}
