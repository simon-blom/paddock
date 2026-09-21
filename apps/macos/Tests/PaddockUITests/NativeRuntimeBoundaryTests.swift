import AppKit
import CoreGraphics
import Foundation
import PaddockClient
import Testing

@testable import PaddockConversationCore
@testable import PaddockNativeMarkdown
@testable import PaddockStudio

@Suite("Native runtime boundary", .serialized)
struct NativeRuntimeBoundaryTests {
  @Test func recordingHeaderPreservesExactFinalPCMSize() {
    let bytes = 16000 * 2 * 3 + 514
    let header = NativeAudioCapture.wavHeader(bytes: bytes)
    #expect(header.count == 44)
    #expect(String(decoding: header.prefix(4), as: UTF8.self) == "RIFF")
    header.withUnsafeBytes { data in
      #expect(
        UInt32(littleEndian: data.loadUnaligned(fromByteOffset: 4, as: UInt32.self)) == bytes + 36)
      #expect(
        UInt32(littleEndian: data.loadUnaligned(fromByteOffset: 24, as: UInt32.self)) == 16000)
      #expect(
        UInt32(littleEndian: data.loadUnaligned(fromByteOffset: 40, as: UInt32.self)) == bytes)
    }
  }
  @Test func selectedPDFPagesRasterizeWithoutAViewer() async throws {
    let bytes = NSMutableData()
    let consumer = try #require(CGDataConsumer(data: bytes))
    var page = CGRect(x: 0, y: 0, width: 120, height: 160)
    let context = try #require(CGContext(consumer: consumer, mediaBox: &page, nil))
    for _ in 0..<3 {
      context.beginPDFPage(nil)
      context.fill(page)
      context.endPDFPage()
    }
    context.closePDF()
    let parts = try await NativePDFInput.shared.parts(
      bytes as Data, metadata: ["name": .string("Pages.pdf"), "pageRange": .string("2-3")])
    #expect(parts.count == 4)
    #expect(parts[0]["text"]?.string == "[Pages.pdf, page 2]")
    #expect(parts[1]["image_url"]?.string?.hasPrefix("data:image/jpeg;base64,") == true)
    await #expect(throws: (any Error).self) {
      try await NativePDFInput.shared.parts(bytes as Data, metadata: ["pageRange": .string("4-5")])
    }
  }
  @Test @MainActor func openingAWorkspaceDoesNotCreateWebKit() async {
    let workspace = StudioWorkspace(client: NoCore())
    #expect(!workspace.hasWebViewer)
    await workspace.setDark(true)
    #expect(!workspace.hasWebViewer)
    await workspace.shutdown()
    #expect(!workspace.hasWebViewer)
  }
  @Test func nativeProjectionDecodesIntoExistingSwiftViews() async throws {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in
    }
    let data = await runtime.uiFixture()
    let decoded = try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(data))
    #expect(decoded.nativeTranscript?.messages.count == 2)
    #expect(decoded.nativeTranscript?.messages.last?.toolCalls?.count == 1)
    #expect(decoded.nativeTranscript?.messages.last?.documentResult?.pages.count == 1)
    #expect(decoded.audio?.mode == "dictate")
    #expect(decoded.library?.page == 0)
    await runtime.close()
  }
  @Test func syntaxHighlightingIsNativeAndNeverChangesSource() async throws {
    for dark in [false, true] {
      for language in ["swift", "python", "javascript", "json", "sql", "unknown"] {
        let source = "let emoji = \"🦊 // not a comment\"\n# comment\n42 < 80\n"
        let rendered = try await NativeCodeHighlighter.shared.highlight(
          source, language: language, dark: dark)
        #expect(String(rendered.characters) == source)
      }
    }
  }
  private struct NoCore: ManagerLoading {
    func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
  }
}

extension NativeStudioRuntime {
  fileprivate func uiFixture() -> O {
    models = [
      [
        "id": .string("model"), "title": .string("Model"), "vendor": .string(""),
        "provider": .string("Local"), "status": .string("ok"), "port": .number(12481),
        "kind": .string("chat"),
      ],
      [
        "id": .string("speech"), "title": .string("Speech"), "vendor": .string(""),
        "provider": .string("Local"), "status": .string("ok"), "port": .number(11540),
        "kind": .string("transcriber"),
      ],
    ]
    caps = ["model": ["max_ctx": .number(4096), "reasoning": .string("toggle")]]
    document = try! .init(fields: [
      "id": .string("native-shape"), "title": .string("Native"), "model": .string("model"),
      "systemPrompt": .string(""),
      "messages": .array([
        .object([
          "id": .string("u"), "role": .string("user"), "parentId": .null,
          "content": .array([.object(["type": .string("text"), "text": .string("Hi")])]),
        ]),
        .object([
          "id": .string("a"), "role": .string("assistant"), "model": .string("model"),
          "parentId": .string("u"),
          "content": .array([.object(["type": .string("text"), "text": .string("Hello")])]),
          "error": .null,
          "toolCalls": .array([
            .object([
              "id": .string("call"), "name": .string("read"), "serverLabel": .string("files"),
              "status": .string("completed"),
            ])
          ]), "ocr": .object(["model": .string("model")]),
        ]),
      ]), "leafId": .string("a"),
    ])
    draft = false
    revision = 1
    return presentation()
  }
}
