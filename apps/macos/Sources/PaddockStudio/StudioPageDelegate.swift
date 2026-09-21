import AVFoundation
import AppKit
import WebKit

/// Web content never chooses local paths. Native panels mediate exports;
/// generated sandboxed frames cannot request files, permissions or dialogs.
@MainActor final class StudioPageDelegate: NSObject, WKUIDelegate, WKDownloadDelegate {
  weak var owner: StudioWorkspace?
  init(owner: StudioWorkspace) { self.owner = owner }
  private func trusted(_ frame: WKFrameInfo) -> Bool {
    frame.isMainFrame && owner?.isLocal(frame.request.url) == true
  }
  func webView(
    _ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable ([URL]?) -> Void
  ) {
    // Attachments belong to Swift's composer, not an HTML file input.
    completionHandler(nil)
  }
  func webView(
    _ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
    initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
    decisionHandler: @escaping @MainActor @Sendable (WKPermissionDecision) -> Void
  ) {
    // Lector, Scriptor and Traverse never own capture. Even a trusted viewer
    // is denied; the only microphone path is AVFoundation in Swift.
    decisionHandler(.deny)
  }
  func webView(
    _ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
  ) {
    guard trusted(frame), let window = webView.window else {
      completionHandler(false)
      return
    }
    let alert = NSAlert()
    alert.messageText = String(message.prefix(2000))
    alert.addButton(withTitle: "Continue")
    alert.addButton(withTitle: "Cancel")
    alert.beginSheetModal(for: window) { completionHandler($0 == .alertFirstButtonReturn) }
  }
  func webView(
    _ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
    initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable () -> Void
  ) {
    guard trusted(frame), let window = webView.window else {
      completionHandler()
      return
    }
    let alert = NSAlert()
    alert.messageText = String(message.prefix(2000))
    alert.addButton(withTitle: "OK")
    alert.beginSheetModal(for: window) { _ in completionHandler() }
  }
  func webView(
    _ webView: WKWebView, runJavaScriptTextInputPanelWithPrompt prompt: String,
    defaultText: String?, initiatedByFrame frame: WKFrameInfo,
    completionHandler: @escaping @MainActor @Sendable (String?) -> Void
  ) {
    guard trusted(frame), let window = webView.window else {
      completionHandler(nil)
      return
    }
    let alert = NSAlert()
    alert.messageText = String(prompt.prefix(2000))
    let field = NSTextField(string: defaultText ?? "")
    field.frame = NSRect(x: 0, y: 0, width: 320, height: 24)
    alert.accessoryView = field
    alert.addButton(withTitle: "Save")
    alert.addButton(withTitle: "Cancel")
    alert.beginSheetModal(for: window) {
      completionHandler($0 == .alertFirstButtonReturn ? field.stringValue : nil)
    }
  }
  func webView(
    _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
    for action: WKNavigationAction, windowFeatures: WKWindowFeatures
  ) -> WKWebView? {
    if trusted(action.sourceFrame), action.navigationType == .linkActivated,
      let url = action.request.url, ["https", "http"].contains(url.scheme)
    {
      NSWorkspace.shared.open(url)
    }
    return nil
  }
  func download(
    _ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String,
    completionHandler: @escaping @MainActor @Sendable (URL?) -> Void
  ) {
    guard let window = owner?.presentationWindow ?? owner?.viewerWindow else {
      completionHandler(nil)
      return
    }
    let panel = NSSavePanel()
    panel.nameFieldStringValue = URL(fileURLWithPath: suggestedFilename).lastPathComponent
    panel.beginSheetModal(for: window) { completionHandler($0 == .OK ? panel.url : nil) }
  }
}

/// Intercept real AppKit file drops before WebKit can navigate to file:// or
/// give the page its own competing attachment tray.
@MainActor final class StudioWebView: WKWebView {
  var onFiles: (([URL]) -> Void)?
  private func files(_ sender: any NSDraggingInfo) -> [URL] {
    sender.draggingPasteboard.readObjects(
      forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
  }
  override func draggingEntered(_ sender: any NSDraggingInfo) -> NSDragOperation {
    files(sender).isEmpty ? super.draggingEntered(sender) : .copy
  }
  override func draggingUpdated(_ sender: any NSDraggingInfo) -> NSDragOperation {
    files(sender).isEmpty ? super.draggingUpdated(sender) : .copy
  }
  override func performDragOperation(_ sender: any NSDraggingInfo) -> Bool {
    let urls = files(sender)
    guard !urls.isEmpty else { return super.performDragOperation(sender) }
    onFiles?(urls)
    return true
  }
}
