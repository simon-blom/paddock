import AppKit
import Foundation
import Network
import SwiftUI
import Synchronization
import Testing
import WebKit

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Isolated native HTML artifacts", .serialized) @MainActor
struct ArtifactPreviewTests {
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_TEST_ARTIFACT_ORIGIN"] != nil))
  func installedNativeAPIHostsTheCredentialFreePreview() async throws {
    let origin = try #require(
      URL(string: ProcessInfo.processInfo.environment["PADDOCK_TEST_ARTIFACT_ORIGIN"] ?? ""))
    let session = ArtifactPreviewSession()
    defer { session.close() }
    await session.render(
      html:
        "<h1>Paddock artifact</h1><script>parent.postMessage({paddockArtifactMissing:{failed:['installed-host-executed']}},'*')</script>",
      origin: origin, allowImages: false, dark: false)
    try await wait { session.failed.contains("installed-host-executed") || session.error != nil }
    #expect(session.error == nil)
    #expect(session.failed == ["installed-host-executed"])
  }
  @Test func sandboxExecutesPageButRejectsCredentialsBridgeNetworkAndNavigation() async throws {
    _ = NSApplication.shared
    let server = try await ArtifactFixtureServer.start()
    defer { server.close() }
    let session = ArtifactPreviewSession()
    defer { session.close() }
    let html = """
      <!doctype html><html><body><button id="count">Count</button><div id="value">0</div>
      <script>
      const results=[]; let n=0;
      count.onclick=()=>value.textContent=String(++n); count.click();
      results.push(value.textContent==='1'?'interactive':'NO-INTERACTION');
      try{void parent.document.body;results.push('PARENT-LEAK')}catch{results.push('parent-blocked')}
      try{void document.cookie;results.push('COOKIE-LEAK')}catch{results.push('cookie-blocked')}
      try{window.webkit.messageHandlers.artifactStatus.postMessage({ready:true});results.push('BRIDGE-LEAK')}catch{results.push('bridge-blocked')}
      try{localStorage.setItem('test','works');results.push(localStorage.getItem('test')==='works'?'memory-storage':'NO-STORAGE')}catch{}
      fetch('\(server.origin)/probe-fetch').catch(()=>{});
      const im=new Image();im.src='\(server.origin)/probe-image';document.body.append(im);
      const script=document.createElement('script');script.src='\(server.origin)/probe-script';document.body.append(script);
      const child=document.createElement('iframe');child.src='\(server.origin)/probe-frame';document.body.append(child);
      window.open('\(server.origin)/probe-popup');
      setTimeout(()=>parent.postMessage({paddockArtifactMissing:{failed:results}},'*'),700);
      </script></body></html>
      """
    await session.render(html: html, origin: server.origin, allowImages: false, dark: false)
    try await wait { session.failed.contains("interactive") || session.error != nil }
    #expect(session.error == nil, "\(session.error ?? "")")
    #expect(
      Set(session.failed) == [
        "interactive", "parent-blocked", "cookie-blocked", "bridge-blocked", "memory-storage",
      ])
    let view = try #require(session.webView)
    #expect(!view.configuration.websiteDataStore.isPersistent)
    #expect(await view.configuration.websiteDataStore.httpCookieStore.allCookies().isEmpty)
    #expect(
      try await view.evaluateJavaScript("document.querySelector('iframe').sandbox.value") as? String
        == "allow-scripts")
    #expect(
      try await view.evaluateJavaScript("document.querySelector('iframe').contentDocument === null")
        as? Bool == true)
    #expect(
      try await view.evaluateJavaScript("typeof window.paddockArtifactMount") as? String
        == "undefined", "Bridge must not exist in the page world")
    #expect(
      server.requests.allSatisfy { ["/native-artifact-host", "/artifact-frame"].contains($0.path) })
    #expect(
      server.requests.allSatisfy {
        !$0.headers.lowercased().contains("cookie:")
          && !$0.headers.lowercased().contains("authorization:")
      })

    // A source edit/new version gets a fresh store; only the selected body is
    // delivered, and explicit picture consent does not unlock fetch or scripts.
    let previous = view
    await session.render(
      html:
        "<script>fetch('\(server.origin)/probe-again').catch(()=>{});setTimeout(()=>parent.postMessage({paddockArtifactMissing:{failed:['new-version']}},'*'),100),0</script><h1>Next version</h1>",
      origin: server.origin, allowImages: true, dark: true)
    try await wait { session.failed.contains("new-version") || session.error != nil }
    #expect(session.error == nil)
    #expect(session.webView !== previous)
    #expect(
      previous.superview == nil && previous.navigationDelegate == nil && previous.uiDelegate == nil)
    #expect(server.requests.contains { $0.path == "/artifact-frame?img=1" })
    #expect(!server.requests.contains { $0.path.hasPrefix("/probe") })
    session.close()
    #expect(session.webView == nil && !session.loading)
  }

  @Test func nativeArtifactUsesMinimalScrollbarWithoutOpeningItsSandbox() async throws {
    let server = try await ArtifactFixtureServer.start()
    defer { server.close() }
    let session = ArtifactPreviewSession()
    defer { session.close() }
    for dark in [false, true] {
      let html = """
        <div id="fixture" style="overflow:auto;width:200px;height:100px"><div style="width:900px;height:900px">Content</div></div>
        <script>setTimeout(()=>{
          const e=document.getElementById('fixture'), t=getComputedStyle(e,'::-webkit-scrollbar-thumb');
          parent.postMessage({paddockArtifactMissing:{failed:[getComputedStyle(e,'::-webkit-scrollbar').width,t.borderLeftWidth,t.backgroundColor]}},'*');
        },100)</script>
        """
      await session.render(html: html, origin: server.origin, allowImages: false, dark: dark)
      try await wait { !session.failed.isEmpty || session.error != nil }
      #expect(session.error == nil)
      #expect(
        session.failed == [
          "10px", "3px", dark ? "rgba(255, 255, 255, 0.28)" : "rgba(0, 0, 0, 0.28)",
        ])
      let view = try #require(session.webView)
      #expect(
        try await view.evaluateJavaScript(
          "document.querySelector('iframe').contentDocument === null") as? Bool == true)
    }
  }

  @Test func compareSessionsDoNotShareStorageOrBodiesAndCancelCleanly() async throws {
    let server = try await ArtifactFixtureServer.start()
    defer { server.close() }
    let left = ArtifactPreviewSession()
    let right = ArtifactPreviewSession()
    defer {
      left.close()
      right.close()
    }
    async let l: Void = left.render(
      html:
        "<script>localStorage.setItem('lane','left');parent.postMessage({paddockArtifactMissing:{failed:['left']}},'*')</script>",
      origin: server.origin, allowImages: false, dark: false)
    async let r: Void = right.render(
      html:
        "<script>parent.postMessage({paddockArtifactMissing:{failed:[localStorage.getItem('lane') || 'right-isolated']}},'*')</script>",
      origin: server.origin, allowImages: false, dark: false)
    _ = await (l, r)
    try await wait { left.failed.contains("left") && right.failed.contains("right-isolated") }
    #expect(
      left.webView?.configuration.websiteDataStore !== right.webView?.configuration.websiteDataStore
    )
    let pending = Task {
      await left.render(
        html: "<h1>Cancelled</h1>", origin: server.origin, allowImages: false, dark: false)
    }
    pending.cancel()
    await pending.value
    #expect(left.webView == nil)
    #expect(right.webView != nil)
  }

  @Test func policyIsAnExactCredentialFreeAllowlist() throws {
    for raw in [
      "https://example.com", "http://localhost:1234", "http://user:pass@127.0.0.1:1234",
      "http://127.0.0.1:1234/api", "http://127.0.0.1:1234?secret=1",
    ] {
      #expect(throws: (any Error).self) {
        try ArtifactPreviewPolicy(origin: URL(string: raw)!, allowImages: false)
      }
    }
    for images in [false, true] {
      let policy = try ArtifactPreviewPolicy(
        origin: URL(string: "http://127.0.0.1:1234")!, allowImages: images)
      let rules =
        try JSONSerialization.jsonObject(with: Data(policy.rulesJSON.utf8)) as! [[String: Any]]
      #expect((rules.first?["action"] as? [String: String])?["type"] == "block")
      let network = rules.filter {
        (($0["trigger"] as? [String: Any])?["url-filter"] as? String) == "^https://"
      }
      #expect(network.count == (images ? 1 : 0))
      if images {
        #expect(
          (network[0]["trigger"] as? [String: Any])?["resource-type"] as? [String] == ["image"])
      }
      #expect(!policy.rulesJSON.contains("cookie"))
    }
  }

  @Test func nativePreviewMountsInsideItsBoundsInLightAndDarkAndTearsDown() async throws {
    _ = NSApplication.shared
    let server = try await ArtifactFixtureServer.start()
    defer { server.close() }
    for dark in [false, true] {
      let html =
        "<style>body{font:18px system-ui;padding:24px;background:#fff;color:#111}button{font:inherit;padding:12px;border-radius:18px}</style><h1>Interactive artifact</h1><p>HTML preview inside native controls.</p><button onclick=\"this.textContent='Clicked'\">Try it</button>"
      let controller = NSHostingController(
        rootView: NativeHTMLArtifact(html: html, origin: server.origin)
          .padding(12).frame(width: 640, height: 480).environment(
            \.colorScheme, dark ? .dark : .light))
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 640, height: 480),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentViewController = controller
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      try await wait { self.webViews(controller.view).first != nil }
      let preview = try #require(webViews(controller.view).first)
      try await wait { !preview.isLoading }
      controller.view.layoutSubtreeIfNeeded()
      #expect(
        preview.bounds.width >= 600 && preview.bounds.height >= 400,
        "Native controls must not collapse the page viewport")
      #expect(controller.view.bounds.contains(preview.convert(preview.bounds, to: controller.view)))
      try await Task.sleep(for: .milliseconds(350))
      let image = try await preview.takeSnapshot(configuration: nil)
      #expect(image.size.width >= 600 && image.size.height >= 400)
      // Release the SwiftUI tree, which must close and detach its session.
      window.contentViewController = NSViewController()
      try await wait { preview.navigationDelegate == nil }
      #expect(preview.superview == nil && preview.uiDelegate == nil)
    }
  }

  private func webViews(_ root: NSView) -> [WKWebView] {
    (root as? WKWebView).map { [$0] } ?? root.subviews.flatMap(webViews)
  }

  private func wait(_ condition: () -> Bool) async throws {
    for _ in 0..<200 {
      if condition() { return }
      try await Task.sleep(for: .milliseconds(50))
    }
    Issue.record("Preview did not reach the expected state")
  }
}

/// Loopback-only HTTP fixture. Serves the production Rust HTML/CSP files, not
/// an easier synthetic sandbox; every attempted HTTP request is recorded.
private final class ArtifactFixtureServer: @unchecked Sendable {
  struct Request: Sendable {
    let path: String
    let headers: String
  }
  private struct State {
    var requests: [Request] = []
    var connections: [NWConnection] = []
  }
  private let state = Mutex(State())
  private let listener: NWListener
  private let queue = DispatchQueue(label: "paddock.artifact.fixture")
  private let host: String, frame: String, csp: String, imageCSP: String, hostCSP: String
  var origin: URL { URL(string: "http://127.0.0.1:\(listener.port!.rawValue)")! }
  var requests: [Request] { state.withLock { $0.requests } }
  private init() throws {
    var repo = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { repo.deleteLastPathComponent() }
    func read(_ name: String) throws -> String {
      try String(contentsOf: repo.appending(path: name), encoding: .utf8)
    }
    host = try read("crates/paddock-desktop/src/native-artifact-host.html")
    frame = try read("crates/paddock-manager/src/artifact-frame.html")
    let rust = try read("crates/paddock-manager/src/artifacts.rs")
    func policy(_ name: String) -> String {
      let after = rust.components(separatedBy: "const \(name): &str = \"")[1].components(
        separatedBy: "\";")[0]
      return after.replacingOccurrences(of: "\\", with: "").split(whereSeparator: \.isWhitespace)
        .joined(separator: " ")
    }
    csp = policy("FRAME_CSP")
    imageCSP = policy("FRAME_CSP_IMG")
    let desktop = try read("crates/paddock-desktop/src/studio.rs")
    hostCSP =
      desktop.components(separatedBy: "(header::CONTENT_SECURITY_POLICY, \"")[1].components(
        separatedBy: "\"")[0]
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
    listener = try NWListener(using: parameters, on: .any)
  }
  static func start() async throws -> ArtifactFixtureServer {
    let server = try ArtifactFixtureServer()
    try await withCheckedThrowingContinuation {
      (continuation: CheckedContinuation<Void, any Error>) in
      server.listener.stateUpdateHandler = { state in
        switch state {
        case .ready:
          server.listener.stateUpdateHandler = nil
          continuation.resume()
        case .failed(let error):
          server.listener.stateUpdateHandler = nil
          continuation.resume(throwing: error)
        default: break
        }
      }
      server.listener.newConnectionHandler = { [weak server] connection in
        guard let server else {
          connection.cancel()
          return
        }
        server.state.withLock { $0.connections.append(connection) }
        connection.start(queue: server.queue)
        server.receive(connection, accumulated: Data())
      }
      server.listener.start(queue: server.queue)
    }
    return server
  }
  private func receive(_ connection: NWConnection, accumulated: Data) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 16384) {
      [weak self] bytes, _, done, error in
      guard let self else {
        connection.cancel()
        return
      }
      var data = accumulated
      if let bytes { data.append(bytes) }
      guard data.count <= 32768 else {
        connection.cancel()
        return
      }
      let header = String(decoding: data, as: UTF8.self)
      if !header.contains("\r\n\r\n") {
        if done || error != nil {
          connection.cancel()
        } else {
          self.receive(connection, accumulated: data)
        }
        return
      }
      let path = header.split(separator: " ").dropFirst().first.map(String.init) ?? ""
      self.state.withLock { $0.requests.append(Request(path: path, headers: header)) }
      let body =
        path == "/native-artifact-host"
        ? self.host : path.hasPrefix("/artifact-frame") ? self.frame : "FORBIDDEN"
      let csp =
        path == "/native-artifact-host"
        ? self.hostCSP : path.contains("img=1") ? self.imageCSP : self.csp
      let content = Data(body.utf8)
      let response =
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Security-Policy: \(csp)\r\nPermissions-Policy: camera=(), microphone=(), geolocation=(), clipboard-read=(), clipboard-write=()\r\nCache-Control: no-store\r\nContent-Length: \(content.count)\r\nConnection: close\r\n\r\n"
      connection.send(
        content: Data(response.utf8) + content,
        completion: .contentProcessed { _ in connection.cancel() })
    }
  }
  func close() {
    listener.cancel()
    state.withLock { state in
      for connection in state.connections { connection.cancel() }
      state.connections = []
    }
  }
}
