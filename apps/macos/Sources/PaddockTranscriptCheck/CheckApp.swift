import AppKit
import Foundation
import PaddockClient
import PaddockTranscript
import WebKit

/// Development-only functional gate, running a real AppKit event loop. Never
/// linked into PaddockMac; no core, model, user database or privileged bridge.
@main enum TranscriptCheckApp {
  @MainActor static func main() {
    let app = NSApplication.shared
    let lifecycle = CheckLifecycle()
    app.delegate = lifecycle
    app.setActivationPolicy(.regular)
    withExtendedLifetime(lifecycle) { app.run() }
  }
}
@MainActor final class CheckLifecycle: NSObject, NSApplicationDelegate {
  private var window: NSWindow?
  func applicationDidFinishLaunching(_ notification: Notification) {
    Task {
      do {
        try await run()
        print(
          "PASS: native transcript streaming, recovery, terminal text, selection, math, Mermaid, theme and navigation isolation"
        )
        exit(0)
      } catch {
        FileHandle.standardError.write(Data("FAIL: \(error)\n".utf8))
        exit(1)
      }
    }
  }
  private func run() async throws {
    guard CommandLine.arguments.count == 2 else {
      throw Failure.check("Supply generated StudioRenderer assets")
    }
    let session = TranscriptSession(assetsRoot: URL(fileURLWithPath: CommandLine.arguments[1]))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 900, height: 800),
      styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
    self.window = window
    window.isReleasedWhenClosed = false
    window.title = "Paddock transcript checks - synthetic content"
    window.contentView = session.webView
    window.center()
    window.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
    defer {
      session.close()
      window.close()
    }
    try await until("module readiness") {
      if let error = session.error { throw Failure.check(error) }
      return session.ready
    }
    try await until("foreground eligibility") { window.occlusionState.contains(.visible) }
    var document = try JSONDecoder().decode(
      ChatDocument.self,
      from: Data(
        #"{"id":"test","title":"Test","model":"fixture","messages":[{"id":"u","role":"user","content":[{"type":"text","text":"Select this earlier message"}]},{"id":"a","role":"assistant","content":[{"type":"text","text":""}],"streaming":true}]}"#
          .utf8))
    await session.show(document)
    func deltas(_ text: String) throws -> [ChatDelta] {
      let data = try JSONSerialization.data(withJSONObject: [
        ["kind": "text", "output_index": 0, "content_index": 0, "delta": text]
      ])
      return try ManagerWire.decode([ChatDelta].self, from: data)
    }
    func project(_ text: String) {
      document.messages[1].content = [.init(type: "text", text: text)]
      document.messages[1].nativeTextParts = ["0000000000:0000000000": text]
    }
    let first = "**Hello** å😀\n\n```swift\nlet x = 1\n```"
    project(first)
    await session.append(try deltas(first), source: document)
    try await js(
      session, "document.querySelector('[data-message-id=\"a\"] strong')?.textContent === 'Hello'",
      "rich streamed Markdown")
    _ = try await session.webView.evaluateJavaScript(
      "const r = document.createRange(); r.selectNodeContents(document.querySelector('.user-text')); getSelection().removeAllRanges(); getSelection().addRange(r)"
    )
    let tail = "\n\nRecovered tail"
    project(first + tail)
    await session.append(try deltas(tail), source: document)
    try await js(
      session, "getSelection().toString() === 'Select this earlier message'",
      "selection survives a delta")
    session.reload()
    try await until("reload readiness") { session.ready }
    let next = "\n\nAfter reload"
    project(first + tail + next)
    await session.append(try deltas(next), source: document)
    try await js(
      session,
      "document.querySelector('.assistant').textContent.includes('Recovered tail') && document.querySelector('.assistant').textContent.includes('After reload')",
      "stream prefix and tail recovery")
    document.messages[1].streaming = false
    document.messages[1].content[0].text =
      first + tail + next
      + "\n\nTerminal-only tail\n\n$$E=mc^2$$\n\n```mermaid\nflowchart LR\n A[Shared renderer] --> B[Web]\n A --> C[Swift]\n```"
    document.messages[1].nativeTextParts = nil
    await session.show(document)
    try await js(
      session,
      "document.querySelector('.assistant').textContent.includes('Terminal-only tail') && document.querySelector('.katex') !== null && document.querySelector('.assistant svg') !== null",
      "terminal Markdown, math and Mermaid")
    await session.setDark(true)
    try await js(
      session,
      "document.documentElement.classList.contains('dark') && typeof window.webkit?.messageHandlers === 'undefined'",
      "theme and no privileged bridge")
    guard !session.webView.isInspectable else {
      throw Failure.check("Product renderer exposes Inspector")
    }
    let before = session.webView.url
    session.webView.load(URLRequest(url: URL(string: "https://example.com/")!))
    try await Task.sleep(for: .milliseconds(200))
    guard session.webView.url == before else {
      throw Failure.check("External navigation escaped the bundle")
    }
    guard session.error == nil else { throw Failure.check(session.error ?? "Renderer failed") }
  }
  private func js(_ session: TranscriptSession, _ expression: String, _ label: String) async throws
  {
    try await until(label) {
      try await session.webView.evaluateJavaScript(expression) as? Bool == true
    }
  }
  private func until(_ label: String, _ predicate: () async throws -> Bool) async throws {
    let deadline = Date().addingTimeInterval(20)
    while try await !predicate() {
      if Date() > deadline { throw Failure.check("Timed out: \(label)") }
      try await Task.sleep(for: .milliseconds(20))
    }
  }
  enum Failure: Error { case check(String) }
}
