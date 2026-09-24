import AppKit
import PaddockClient
import PaddockConversationCore
import WebKit

/// The authenticated document/graph WKWebView (never reused for artifacts).
/// Its bridge exposes no chat commands,
/// model selection or history; browser media capture is denied. Its viewer-only
/// module graph rejects the retired Studio runtime. HTTP authorization still
/// uses the existing private-session cookie, not a per-document capability.
@MainActor final class StudioViewerHost: NSObject, WKNavigationDelegate {
  let webView: WKWebView
  private let page: StudioPageDelegate
  private weak var owner: StudioWorkspace?
  private let role: NativeViewerRole
  // Admission supplies the graph before its new user turn is persisted. The
  // startup task must not overwrite that source with the preceding document.
  var preparedForAdmission = false
  private var host: StudioHost?
  private var loaded = false, closed = false
  init(owner: StudioWorkspace, role: NativeViewerRole) {
    self.owner = owner
    self.role = role
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    configuration.mediaTypesRequiringUserActionForPlayback = .all
    let view = StudioWebView(frame: .zero, configuration: configuration)
    webView = view
    page = StudioPageDelegate(owner: owner)
    super.init()
    view.onFiles = { [weak owner] in owner?.addFiles($0) }
    view.navigationDelegate = self
    view.uiDelegate = page
    view.underPageBackgroundColor = .clear
  }
  func start(host: StudioHost) async throws {
    guard !closed, host.isValidPrivateHost else { throw ManagerError.core("Invalid viewer host") }
    self.host = host
    guard
      let cookie = HTTPCookie.cookies(
        withResponseHeaderFields: [
          "Set-Cookie": "\(host.cookieName)=\(host.session); Path=/; HttpOnly; SameSite=Strict"
        ], for: host.origin
      ).first
    else {
      throw ManagerError.core("Could not secure the viewer")
    }
    await webView.configuration.websiteDataStore.httpCookieStore.setCookie(cookie)
    guard !closed else { return }
    webView.load(URLRequest(url: host.origin.appending(path: "studio")))
    try await waitUntilReady()
  }
  func waitUntilReady() async throws {
    for _ in 0..<400 {
      try Task.checkCancellation()
      guard !closed else { throw CancellationError() }
      if loaded { return }
      try await Task.sleep(for: .milliseconds(50))
    }
    throw ManagerError.core("The document viewer did not open")
  }
  @discardableResult func update(_ fields: [String: ConversationValue], dark: Bool) async throws
    -> String
  {
    guard !closed, loaded else { return "" }
    let bytes = try JSONEncoder().encode(role.project(fields))
    guard bytes.count <= 8 * 1024 * 1024 else {
      throw ManagerError.core("The viewer projection is too large")
    }
    let result = try await webView.callAsyncJavaScript(
      "return await window.paddockViewer.update(JSON.parse(json), dark)",
      arguments: ["json": String(decoding: bytes, as: UTF8.self), "dark": dark], in: nil,
      contentWorld: .page)
    return result as? String ?? ""
  }
  func action(_ name: String) async throws {
    guard loaded, !closed, ["info", "download"].contains(name) else {
      throw ManagerError.core("Open a document first")
    }
    _ = try await webView.callAsyncJavaScript(
      "window.paddockViewer.action(name)", arguments: ["name": name], in: nil, contentWorld: .page)
  }
  func close() {
    guard !closed else { return }
    closed = true
    loaded = false
    webView.stopLoading()
    webView.navigationDelegate = nil
    webView.uiDelegate = nil
    webView.removeFromSuperview()
    // Navigating away releases workers, graphics and sockets. No chat/mic
    // lifetime is tied to this page and no shared WKProcessPool is retained.
    webView.loadHTMLString("", baseURL: nil)
  }
  func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    Task {
      do {
        _ = try await webView.callAsyncJavaScript(
          "for(let i=0;i<200;i++){if(window.paddockViewer)return;await new Promise(r=>setTimeout(r,25))}throw new Error('Viewer did not initialize')",
          arguments: [:], in: nil, contentWorld: .page)
        if !closed { loaded = true }
      } catch { if !closed { owner?.error = error.localizedDescription } }
    }
  }
  func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
    loaded = false
    owner?.error =
      "The document viewer stopped. Close and reopen it; your conversation and microphone are unaffected."
  }
  func webView(
    _ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
    decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
  ) {
    guard let url = action.request.url else {
      decisionHandler(.cancel)
      return
    }
    let trusted =
      action.sourceFrame.isMainFrame && owner?.isLocal(action.sourceFrame.request.url) == true
    if trusted, action.shouldPerformDownload, owner?.isLocal(url) == true || url.scheme == "blob" {
      decisionHandler(.download)
      return
    }
    if action.targetFrame?.isMainFrame != false, owner?.isLocal(url) == true, url.path == "/studio"
    {
      decisionHandler(.allow)
      return
    }
    if trusted, action.navigationType == .linkActivated, ["http", "https"].contains(url.scheme) {
      NSWorkspace.shared.open(url)
    }
    decisionHandler(.cancel)
  }
  func webView(
    _ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload
  ) { download.delegate = page }
}
