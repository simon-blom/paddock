import AppKit
import Foundation
import Observation
import WebKit

/// Native shell for untrusted generated pages. It never receives a StudioHost
/// session or uses StudioViewerHost, StudioPageDelegate or their authenticated IO.
@MainActor @Observable
public final class ArtifactPreviewSession: NSObject, WKNavigationDelegate, WKUIDelegate {
  public private(set) var webView: WKWebView?
  public private(set) var loading = false
  public private(set) var error: String?
  public private(set) var blocked: [String] = []
  public private(set) var failed: [String] = []
  @ObservationIgnored private var policy: ArtifactPreviewPolicy?
  @ObservationIgnored private var generation = 0
  @ObservationIgnored private var mainAdmission = false
  @ObservationIgnored private var frameAdmission = false
  @ObservationIgnored private var shellReady = false
  @ObservationIgnored private let world = WKContentWorld.world(name: "PaddockArtifactHost")
  @ObservationIgnored private var watchdog: Task<Void, Never>?

  public override init() { super.init() }

  public func render(html: String, origin: URL, allowImages: Bool, dark: Bool) async {
    close()
    generation += 1
    let ticket = generation
    loading = true
    error = nil
    blocked = []
    failed = []
    do {
      guard html.utf8.count <= 4 * 1024 * 1024 else {
        throw PreviewError("Preview is limited to 4 MiB. Source and export remain available.")
      }
      let policy = try ArtifactPreviewPolicy(origin: origin, allowImages: allowImages)
      let rules = try await ArtifactPreviewRules.shared.rules(policy)
      try Task.checkCancellation()
      guard ticket == generation else { return }
      let config = WKWebViewConfiguration()
      config.websiteDataStore = .nonPersistent()  // New, never the document viewer's store.
      config.preferences.javaScriptCanOpenWindowsAutomatically = false
      config.mediaTypesRequiringUserActionForPlayback = .all
      config.userContentController.add(rules)
      let script = try String(
        contentsOf: Bundle.module.url(forResource: "ArtifactPreviewBridge", withExtension: "js")!,
        encoding: .utf8)
      config.userContentController.addUserScript(
        WKUserScript(
          source: script, injectionTime: .atDocumentStart, forMainFrameOnly: true, in: world))
      config.userContentController.add(
        ArtifactPreviewMessages(self), contentWorld: world, name: "artifactStatus")
      let view = ArtifactWebView(frame: .zero, configuration: config)
      view.navigationDelegate = self
      view.uiDelegate = self
      view.underPageBackgroundColor = .clear
      self.policy = policy
      mainAdmission = true
      shellReady = false
      webView = view
      view.load(URLRequest(url: policy.hostURL, cachePolicy: .reloadIgnoringLocalCacheData))
      for _ in 0..<200 {
        try Task.checkCancellation()
        guard ticket == generation else { return }
        if shellReady { break }
        if let error { throw PreviewError(error) }
        try await Task.sleep(for: .milliseconds(50))
      }
      guard shellReady else {
        throw PreviewError("The preview did not load. Source and export remain available.")
      }
      frameAdmission = true
      _ = try await view.callAsyncJavaScript(
        "window.paddockArtifactMount(html, url, dark)",
        arguments: ["html": html, "url": policy.frameURL.absoluteString, "dark": dark], in: nil,
        contentWorld: world)
      guard ticket == generation else { return }
      watchdog = Task { [weak self] in
        try? await Task.sleep(for: .seconds(10))
        guard !Task.isCancelled, let self, self.generation == ticket, self.loading else { return }
        self.fail("The preview did not respond. Reload it or inspect the source.")
      }
    } catch is CancellationError {
      if ticket == generation { close() }
    } catch {
      if ticket == generation { fail(error.localizedDescription) }
    }
  }

  public func close() {
    generation += 1
    watchdog?.cancel()
    watchdog = nil
    loading = false
    shellReady = false
    mainAdmission = false
    frameAdmission = false
    if let view = webView {
      view.stopLoading()
      view.configuration.userContentController.removeScriptMessageHandler(
        forName: "artifactStatus", contentWorld: world)
      view.configuration.userContentController.removeAllUserScripts()
      view.navigationDelegate = nil
      view.uiDelegate = nil
      view.removeFromSuperview()
      view.loadHTMLString("", baseURL: nil)
    }
    webView = nil
    policy = nil
  }
  private func fail(_ message: String) {
    close()
    error = message
  }
  fileprivate func receive(_ message: WKScriptMessage) {
    guard message.webView === webView, message.frameInfo.isMainFrame,
      message.frameInfo.request.url == policy?.hostURL,
      let body = message.body as? [String: Any]
    else { return }
    if body["ready"] as? Bool == true {
      loading = false
      watchdog?.cancel()
      watchdog = nil
    }
    func bounded(_ value: Any?) -> [String] {
      (value as? [String] ?? []).prefix(32).map { String($0.prefix(300)) }
    }
    if body["blocked"] != nil { blocked = bounded(body["blocked"]) }
    if body["failed"] != nil { failed = bounded(body["failed"]) }
  }

  public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    if webView === self.webView, webView.url == policy?.hostURL { shellReady = true }
  }
  public func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: any Error
  ) {
    if webView === self.webView {
      fail("The preview could not load: \(error.localizedDescription)")
    }
  }
  public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
    if webView === self.webView {
      fail("The preview process stopped. Reload it; your source and conversation are preserved.")
    }
  }
  public func webView(
    _ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
    decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
  ) {
    guard webView === self.webView, !action.shouldPerformDownload, let policy else {
      decisionHandler(.cancel)
      return
    }
    if action.targetFrame?.isMainFrame == true, mainAdmission, action.request.url == policy.hostURL
    {
      mainAdmission = false
      decisionHandler(.allow)
      return
    }
    if action.targetFrame?.isMainFrame == false, frameAdmission,
      action.request.url == policy.frameURL
    {
      frameAdmission = false
      decisionHandler(.allow)
      return
    }
    decisionHandler(.cancel)  // No external navigation, downloads, file: or self reloads.
  }
  public func webView(
    _ webView: WKWebView, didReceive challenge: URLAuthenticationChallenge,
    completionHandler:
      @escaping @MainActor @Sendable (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
  ) {
    completionHandler(.cancelAuthenticationChallenge, nil)
  }
  public func webView(
    _ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
    decisionHandler: @escaping @MainActor @Sendable (WKNavigationResponsePolicy) -> Void
  ) {
    guard webView === self.webView, let policy,
      response.response.url == (response.isForMainFrame ? policy.hostURL : policy.frameURL),
      response.response.mimeType == "text/html", response.canShowMIMEType,
      (response.response as? HTTPURLResponse)?.statusCode == 200
    else {
      decisionHandler(.cancel)
      return
    }
    decisionHandler(.allow)
  }
  public func webView(
    _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
    for action: WKNavigationAction, windowFeatures: WKWindowFeatures
  ) -> WKWebView? { nil }
  public func webView(
    _ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable ([URL]?) -> Void
  ) { completionHandler(nil) }
  public func webView(
    _ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
    initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
    decisionHandler: @escaping @MainActor @Sendable (WKPermissionDecision) -> Void
  ) { decisionHandler(.deny) }
  @available(macOS 27.0, *)
  public func webView(
    _ webView: WKWebView, requestGeolocationPermissionFor origin: WKSecurityOrigin,
    initiatedByFrame frame: WKFrameInfo,
    decisionHandler: @escaping @MainActor @Sendable (WKPermissionDecision) -> Void
  ) { decisionHandler(.deny) }
  public func webView(
    _ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable () -> Void
  ) { completionHandler() }
  public func webView(
    _ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
  ) { completionHandler(false) }
  public func webView(
    _ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
    defaultText: String?, initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable (String?) -> Void
  ) { completionHandler(nil) }
}

private struct PreviewError: LocalizedError {
  let message: String
  init(_ message: String) { self.message = message }
  var errorDescription: String? { message }
}
@MainActor private final class ArtifactPreviewMessages: NSObject, WKScriptMessageHandler {
  weak var session: ArtifactPreviewSession?
  init(_ session: ArtifactPreviewSession) { self.session = session }
  func userContentController(
    _ userContentController: WKUserContentController, didReceive message: WKScriptMessage
  ) { session?.receive(message) }
}
@MainActor private final class ArtifactWebView: WKWebView {
  override func draggingEntered(_ sender: any NSDraggingInfo) -> NSDragOperation { [] }
  override func draggingUpdated(_ sender: any NSDraggingInfo) -> NSDragOperation { [] }
  override func performDragOperation(_ sender: any NSDraggingInfo) -> Bool { false }
}

struct ArtifactPreviewPolicy: Hashable, Sendable {
  let hostURL: URL
  let frameURL: URL
  let allowImages: Bool
  init(origin: URL, allowImages: Bool) throws {
    guard origin.scheme == "http", origin.host == "127.0.0.1", let port = origin.port,
      (1...65535).contains(port), origin.user == nil, origin.password == nil, origin.query == nil,
      origin.fragment == nil, ["", "/"].contains(origin.path)
    else { throw PreviewError("Invalid artifact preview origin") }
    hostURL = origin.appending(path: "native-artifact-host")
    frameURL = URL(
      string: origin.appending(path: "artifact-frame").absoluteString
        + (allowImages ? "?img=1" : ""))!
    self.allowImages = allowImages
  }
  var rulesJSON: String {
    var rules: [[String: Any]] = [["trigger": ["url-filter": ".*"], "action": ["type": "block"]]]
    for url in [hostURL, frameURL] {
      rules.append([
        "trigger": [
          "url-filter": "^" + NSRegularExpression.escapedPattern(for: url.absoluteString) + "$",
          "resource-type": ["document"],
        ], "action": ["type": "ignore-previous-rules"],
      ])
    }
    // WebKit's content-rule regex language does not support alternation.
    for scheme in ["data", "blob"] {
      rules.append([
        "trigger": ["url-filter": "^\(scheme):", "resource-type": ["image", "media", "font"]],
        "action": ["type": "ignore-previous-rules"],
      ])
    }
    if allowImages {
      rules.append([
        "trigger": ["url-filter": "^https://", "resource-type": ["image"]],
        "action": ["type": "ignore-previous-rules"],
      ])
    }
    return String(
      decoding: try! JSONSerialization.data(withJSONObject: rules, options: [.sortedKeys]),
      as: UTF8.self)
  }
}

/// Content-rule compilation is shared/coalesced across Compare panes. Keep
/// only a bounded in-memory cache; no per-version disk rule-list growth.
@MainActor private final class ArtifactPreviewRules {
  static let shared = ArtifactPreviewRules()
  var tasks: [ArtifactPreviewPolicy: Task<WKContentRuleList, any Error>] = [:]
  func rules(_ policy: ArtifactPreviewPolicy) async throws -> WKContentRuleList {
    if let task = tasks[policy] { return try await task.value }
    if tasks.count >= 4 { tasks = [:] }
    let task = Task { @MainActor in
      let store = WKContentRuleListStore.default()!
      let id = "paddock-artifact-\(UUID().uuidString)"
      let rules = try await store.compileContentRuleList(
        forIdentifier: id, encodedContentRuleList: policy.rulesJSON)!
      try? await store.removeContentRuleList(forIdentifier: id)
      return rules
    }
    tasks[policy] = task
    do { return try await task.value } catch {
      tasks[policy] = nil
      throw error
    }
  }
}
