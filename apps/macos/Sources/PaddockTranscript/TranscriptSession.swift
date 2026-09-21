import AppKit
import Foundation
import Observation
import PaddockClient
import PaddockWebAssets
import WebKit

/// One document view per visible conversation, not one WebView per message.
/// The renderer receives presentation data only. There is deliberately no
/// WKScriptMessageHandler: rendered content cannot call the core or choose a
/// privileged command. Recovery comes from the native source projection.
@MainActor @Observable
public final class TranscriptSession: NSObject, WKNavigationDelegate {
  public let webView: WKWebView
  public private(set) var error: String?
  public private(set) var ready = false
  private var source: ChatDocument?
  private var revision: UInt64 = 0
  private var renderedRevision: UInt64 = 0
  private var navigationGeneration: UInt64 = 0
  private var sending = false
  private var waiters: [CheckedContinuation<Void, Never>] = []
  private var lastRecovery: Date?
  private var dark = false
  private let assets: BundledAssets
  private var closed = false

  public override convenience init() {
    self.init(assetsRoot: Bundle.main.resourceURL!.appending(path: "StudioRenderer"))
  }
  public init(assetsRoot: URL) {
    assets = BundledAssets(root: assetsRoot, surface: .transcript)
    let config = WKWebViewConfiguration()
    config.websiteDataStore = .nonPersistent()
    config.preferences.javaScriptCanOpenWindowsAutomatically = false
    // WebKit can finish navigation before a custom-scheme module graph has
    // executed. Install the promise before any module, not inside that graph.
    config.userContentController.addUserScript(
      WKUserScript(
        source: """
          window.paddockTranscriptReady = new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('Studio renderer modules did not initialize')), 15000);
            window.paddockTranscriptDidMount = () => { clearTimeout(timer); resolve(); };
            window.addEventListener('error', event => { clearTimeout(timer); reject(new Error(event.message || 'Studio asset failed to load')); }, { once: true });
            window.addEventListener('unhandledrejection', event => { clearTimeout(timer); reject(event.reason); }, { once: true });
            window.addEventListener('securitypolicyviolation', event => { clearTimeout(timer); reject(new Error('Bundled asset blocked by ' + event.effectiveDirective)); }, { once: true });
          });
          """, injectionTime: .atDocumentStart, forMainFrameOnly: true))
    config.setURLSchemeHandler(BundledSchemeHandler(assets: assets), forURLScheme: "paddock-render")
    webView = WKWebView(frame: .zero, configuration: config)
    super.init()
    webView.navigationDelegate = self
    webView.underPageBackgroundColor = .clear
    webView.allowsBackForwardNavigationGestures = false
    webView.isInspectable = false
    if FileManager.default.fileExists(atPath: assetsRoot.appending(path: "index.html").path) {
      reload()
    } else {
      error = "The bundled Studio renderer is missing. Rebuild or reinstall Paddock."
    }
  }
  public func show(_ document: ChatDocument?) async {
    source = document
    revision &+= 1
    guard ready, !closed else { return }
    await acquire()
    defer { release() }
    await replaceLatest(navigation: navigationGeneration)
  }
  public func append(_ events: [ChatDelta], source: ChatDocument) async {
    self.source = source
    revision &+= 1
    let updateRevision = revision
    guard ready, !closed else { return }
    await acquire()
    defer { release() }
    guard ready, !closed, renderedRevision < updateRevision else { return }
    let navigation = navigationGeneration
    if renderedRevision != updateRevision - 1 {
      await replaceLatest(navigation: navigation)
      return
    }
    do {
      let data = try JSONEncoder().encode(events)
      guard data.count <= 256 * 1024 else {
        throw ManagerError.core("Transcript update exceeds its delivery budget.")
      }
      _ = try await webView.callAsyncJavaScript(
        "return await window.paddockTranscript.append(JSON.parse(json), conversationId)",
        arguments: ["json": String(decoding: data, as: UTF8.self), "conversationId": source.id],
        in: nil, contentWorld: .page)
      if navigation == navigationGeneration { renderedRevision = updateRevision }
    } catch {
      if navigation == navigationGeneration, !closed { self.error = error.localizedDescription }
    }
  }
  public func setDark(_ value: Bool) async {
    dark = value
    guard ready, !closed else { return }
    do {
      _ = try await webView.callAsyncJavaScript(
        "window.paddockTranscript.theme(dark)", arguments: ["dark": dark], in: nil,
        contentWorld: .page)
    } catch { self.error = error.localizedDescription }
  }
  private func acquire() async {
    if !sending {
      sending = true
      return
    }
    await withCheckedContinuation { waiters.append($0) }
  }
  private func describe(_ error: any Error) -> String {
    let native = error as NSError
    let detail = native.userInfo["WKJavaScriptExceptionMessage"] as? String
    return detail.map { "Studio renderer: \($0)" } ?? error.localizedDescription
  }
  private func release() {
    if waiters.isEmpty { sending = false } else { waiters.removeFirst().resume() }
  }
  private func replaceLatest(navigation: UInt64) async {
    // A single delivery lane prevents a slow initial snapshot from overwriting
    // newer deltas. Encoding history does not run on the AppKit main thread.
    while !closed, navigation == navigationGeneration {
      let document = source
      let snapshotRevision = revision
      do {
        let json = try await Task.detached(priority: .userInitiated) {
          let data = try JSONEncoder().encode(document)
          guard data.count <= 8 * 1024 * 1024 else {
            throw ManagerError.core("Transcript exceeds the native 8 MiB view budget.")
          }
          return String(decoding: data, as: UTF8.self)
        }.value
        guard !closed, navigation == navigationGeneration else { return }
        _ = try await webView.callAsyncJavaScript(
          "return await window.paddockTranscript.replace(JSON.parse(json), dark)",
          arguments: ["json": json, "dark": dark], in: nil, contentWorld: .page)
        guard !closed, navigation == navigationGeneration else { return }
        renderedRevision = snapshotRevision
        error = nil
        if revision == snapshotRevision { return }
      } catch {
        if navigation == navigationGeneration, !closed { self.error = describe(error) }
        return
      }
    }
  }
  public func reload() {
    guard !closed else { return }
    ready = false
    navigationGeneration &+= 1
    webView.load(URLRequest(url: URL(string: assets.origin + "/index.html")!))
  }
  public func close() {
    closed = true
    ready = false
    source = nil
    navigationGeneration &+= 1
    webView.stopLoading()
    webView.navigationDelegate = nil
  }
  public func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
    let navigation = navigationGeneration
    Task {
      do {
        _ = try await webView.callAsyncJavaScript(
          "return await window.paddockTranscriptReady", arguments: [:], in: nil, contentWorld: .page
        )
        await acquire()
        defer { release() }
        guard !closed, navigation == navigationGeneration else { return }
        await replaceLatest(navigation: navigation)
        if !closed, navigation == navigationGeneration, error == nil { ready = true }
      } catch { if navigation == navigationGeneration, !closed { self.error = describe(error) } }
    }
  }
  public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
    ready = false
    if let lastRecovery, Date().timeIntervalSince(lastRecovery) < 30 {
      error =
        "The renderer stopped repeatedly. Your response is still held by the core; reload the renderer to retry."
      return
    }
    lastRecovery = Date()
    reload()
  }
  public func webView(
    _ webView: WKWebView, didFail navigation: WKNavigation!, withError error: any Error
  ) { self.error = error.localizedDescription }
  public func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: any Error
  ) { self.error = error.localizedDescription }
  public func webView(
    _ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
    decisionHandler: @escaping @MainActor @Sendable (WKNavigationActionPolicy) -> Void
  ) {
    // No arbitrary network navigation, popups, downloads or file URL access.
    // External links stay inert in this first slice; native-confirmed opening
    // can be added separately without exposing a general action bridge.
    let allowed =
      action.navigationType != .linkActivated && action.targetFrame?.isMainFrame == true
      && action.request.url.flatMap(assets.resource(for:)) != nil
    decisionHandler(allowed ? .allow : .cancel)
  }
}
