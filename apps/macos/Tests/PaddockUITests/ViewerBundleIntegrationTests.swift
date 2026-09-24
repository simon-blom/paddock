import AppKit
import CoreGraphics
import Foundation
import Testing
import WebKit

@testable import PaddockClient
@testable import PaddockConversationCore
@testable import PaddockStudio

/// Opt-in, offscreen integration with the actual Rust host and bundled
/// Lector/Traverse WASM. Never reads the user's library or calls a model.
@Suite("Native viewer bundle", .serialized) @MainActor
struct ViewerBundleIntegrationTests {
  typealias V = ConversationValue
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_NATIVE_VIEWER_TEST"] == "1"))
  func pdfAndGraphRenderTogetherAndRestoreQueriesInBothThemes() async throws {
    let dataRoot = ProcessInfo.processInfo.environment["PADDOCK_DATA"] ?? ""
    guard dataRoot.hasPrefix("/tmp/paddock-native-viewers.") else {
      throw ConversationFailure.invalid("Use isolated native-viewers storage")
    }
    _ = NSApplication.shared
    var repo = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { repo.deleteLastPathComponent() }
    let assets = repo.appending(path: "apps/macos/.build/studio-workspace")
    let manager = NativeManager(
      libraryURL: repo.appending(path: "target/debug/libpaddock_desktop.dylib"))
    let workspace = StudioWorkspace(client: manager)
    do {
      await workspace.start()
      try #require(workspace.ready, "\(workspace.error ?? "Native host did not open")")
      let host = try await manager.studio(assets: assets)
      try await exercise(workspace: workspace, host: host, assets: assets)
      await workspace.shutdown()
      await manager.close()
    } catch {
      await workspace.shutdown()
      await manager.close()
      throw error
    }
  }

  private func exercise(workspace: StudioWorkspace, host: StudioHost, assets: URL) async throws {
    let left = StudioViewerHost(owner: workspace, role: .document)
    let right = StudioViewerHost(owner: workspace, role: .graph)
    defer {
      left.close()
      right.close()
    }
    let root = NSView(frame: NSRect(x: 0, y: 0, width: 900, height: 600))
    left.webView.frame = NSRect(x: 0, y: 0, width: 450, height: 600)
    right.webView.frame = NSRect(x: 450, y: 0, width: 450, height: 600)
    root.addSubview(left.webView)
    root.addSubview(right.webView)
    let window = NSWindow(
      contentRect: root.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = root
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    async let leftReady: Void = left.start(host: host)
    async let rightReady: Void = right.start(host: host)
    _ = try await (leftReady, rightReady)

    // Create a tiny real tvdb with the shipping worker, then upload it through
    // the same authenticated attachment route used by the native composer.
    let sessionAsset = try #require(
      FileManager.default.contentsOfDirectory(
        at: assets.appending(path: "assets"), includingPropertiesForKeys: nil
      )
      .first { $0.lastPathComponent.hasPrefix("session-") && $0.pathExtension == "js" })
    let encoded =
      try await right.webView.callAsyncJavaScript(
        """
        const module = await import('/assets/' + asset);
        const Session = Object.values(module).find(x => typeof x === 'function' && x.prototype?.exportTvdb && x.prototype?.seed);
        const s = new Session();
        try {
          await s.open();
          await s.seed("CREATE (a:Person {name:'Alice'})-[:KNOWS]->(b:Person {name:'Bob'})");
          const bytes = await s.exportTvdb();
          let binary = ''; for (const byte of bytes) binary += String.fromCharCode(byte);
          return btoa(binary);
        } finally { s.close(); }
        """, arguments: ["asset": sessionAsset.lastPathComponent], in: nil, contentWorld: .page)
      as? String
    let graphBytes = try #require(encoded.flatMap { Data(base64Encoded: $0) })
    try await upload(graphBytes, id: "fixture-graph", mime: "application/octet-stream", host: host)
    try await upload(pdf(), id: "fixture-pdf", mime: "application/pdf", host: host)
    let fields: [String: V] = [
      "conversationId": .string("fixture-chat"), "visibleGraph": .bool(true),
      "document": .object([
        "id": .string("fixture-chat"), "activeDocId": .string("source"),
        "leafId": .string("source"),
        "messages": .array([
          .object([
            "id": .string("source"), "parentId": .null, "role": .string("user"),
            "createdAt": .number(0),
            "content": .array([
              .object([
                "type": .string("file"), "mime": .string("application/pdf"),
                "attachmentId": .string("fixture-pdf"),
                "name": .string("Fixture.pdf"), "pages": .number(3),
              ])
            ]),
          ])
        ]),
      ]),
      "graphSource": .object([
        "type": .string("graph"), "attachmentId": .string("fixture-graph"),
        "name": .string("Fixture.tvdb"),
      ]),
      "graphHistory": .array([
        .object([
          "id": .string("answer"), "role": .string("assistant"), "model": .string("qwen"),
          "content": .array([]),
          "toolCalls": .array([
            .object([
              "name": .string("graph_query"), "serverLabel": .string("graph"),
              "arguments": .string("{\"cypher\":\"MATCH (n) RETURN n\"}"),
            ])
          ]),
        ])
      ]),
    ]
    for dark in [false, true] {
      async let doc = left.update(fields, dark: dark)
      async let graph = right.update(fields, dark: dark)
      let (_, grounding) = try await (doc, graph)
      #expect(grounding.contains("2 nodes") && grounding.contains("1 edges"))
      try await wait(left.webView, "!!document.querySelector('.native-document-surface canvas')")
      try await wait(
        right.webView,
        "!!document.querySelector('.gp__canvas canvas') && document.querySelectorAll('.gp__chip').length === 2"
      )
      #expect(
        try await right.webView.evaluateJavaScript(
          "document.querySelector('.native-document-surface') === null") as? Bool == true)
      #expect(
        try await left.webView.evaluateJavaScript("document.querySelector('.gp') === null") as? Bool
          == true)
      _ = try await right.webView.evaluateJavaScript(
        "document.querySelectorAll('.gp__chip')[1].click()")
      try await wait(
        right.webView,
        "!document.querySelector('.gp__qerr') && !!document.querySelector('.gp__chip--on:nth-child(2)')"
      )
    }
    left.close()
    #expect(right.webView.navigationDelegate != nil)
    try await wait(right.webView, "!!document.querySelector('.gp__canvas canvas')")
    let reopened = StudioViewerHost(owner: workspace, role: .document)
    defer { reopened.close() }
    reopened.webView.frame = NSRect(x: 0, y: 0, width: 450, height: 600)
    root.addSubview(reopened.webView)
    try await reopened.start(host: host)
    try await reopened.update(fields, dark: true)
    try await wait(reopened.webView, "!!document.querySelector('.native-document-surface canvas')")
    #expect(
      try await right.webView.evaluateJavaScript("document.querySelectorAll('.gp__chip').length")
        as? Int == 2)
  }

  private func wait(_ view: WKWebView, _ expression: String) async throws {
    for _ in 0..<400 {
      if (try? await view.evaluateJavaScript(expression)) as? Bool == true { return }
      try await Task.sleep(for: .milliseconds(50))
    }
    let text = try? await view.evaluateJavaScript("document.body.innerText.slice(0, 1500)")
    throw ConversationFailure.invalid("Viewer check timed out: \(expression). \(text ?? "")")
  }
  private func upload(_ data: Data, id: String, mime: String, host: StudioHost) async throws {
    let session = URLSession(configuration: .ephemeral)
    defer { session.invalidateAndCancel() }
    var request = URLRequest(url: host.origin.appending(path: "api/attachments/\(id)"))
    request.httpMethod = "PUT"
    request.setValue(mime, forHTTPHeaderField: "Content-Type")
    request.setValue("\(host.cookieName)=\(host.session)", forHTTPHeaderField: "Cookie")
    let (_, response) = try await session.upload(for: request, from: data)
    #expect([200, 204].contains((response as? HTTPURLResponse)?.statusCode ?? 0))
  }
  private func pdf() throws -> Data {
    let bytes = NSMutableData()
    let consumer = try #require(CGDataConsumer(data: bytes))
    var rect = CGRect(x: 0, y: 0, width: 120, height: 180)
    let context = try #require(CGContext(consumer: consumer, mediaBox: &rect, nil))
    for _ in 0..<3 {
      context.beginPDFPage(nil)
      context.setFillColor(CGColor(gray: 0.5, alpha: 1))
      context.fill(CGRect(x: 20, y: 20, width: 80, height: 140))
      context.endPDFPage()
    }
    context.closePDF()
    return bytes as Data
  }
}
