import AppKit
import Observation
import WebKit

@MainActor
private final class LabReportProxy: NSObject, WKScriptMessageHandler {
  weak var session: LabSession?
  func userContentController(
    _ controller: WKUserContentController, didReceive message: WKScriptMessage
  ) {
    session?.userContentController(controller, didReceive: message)
  }
}

public enum LabCommand: String, CaseIterable, Sendable {
  case all, markdown, math, mermaid, pdf, docx, traverse, stress, security, reset, theme
  case scale, scalepdf, scalegraph, scaledocx, scalerepeat
  case scalegraphlimit, scalepdfwalk, scalepdfrepeat
  case scalegraphrepeat
  case scalegraphhover
  case scalestream
  case scalestreamrepeat
  case scalestreamlong
}

/// The bridge accepts diagnostic data only, not paths, URLs, code, credentials or core commands.
public struct LabReport: Codable, Sendable {
  public struct Profile: Codable, Sendable {
    public struct Event: Codable, Sendable {
      public let phase: String
      public let at: Double
    }
    public struct Metric: Codable, Sendable {
      public let phase: String
      public let name: String
      public let value: Double
      public let unit: String
    }
    public struct Span: Codable, Sendable {
      public let phase: String
      public let name: String
      public let start: Double
      public let ms: Double
    }
    public struct GraphTrigger: Codable, Sendable {
      public let phase: String
      public let kind: String
      public let start: Double
      public let ms: Double
      public let causes: [String]
      public let full: Bool?
      public let reindex: Bool?
      public let scheduled: Bool?
      public let nodes: Int?
      public let edges: Int?
      var valid: Bool {
        phase.count <= 120 && ["refresh", "process", "interaction"].contains(kind)
          && start.isFinite && start >= 0 && ms.isFinite && ms >= 0
          && causes.count <= 16 && causes.allSatisfy { $0.count <= 120 }
          && (nodes == nil || (0...200_000).contains(nodes!))
          && (edges == nil || (0...1_000_000).contains(edges!))
      }
    }
    public let phase: String
    public let events: [Event]
    public let metrics: [Metric]
    public let spans: [Span]?
    public let graphTriggers: [GraphTrigger]?
    public let graphTriggersDropped: Int?
    var valid: Bool {
      phase.count <= 120 && events.count <= 256 && metrics.count <= 1024
        && (spans?.count ?? 0) <= 512
        && (graphTriggers?.count ?? 0) <= 768
        && (graphTriggers?.allSatisfy { $0.valid } ?? true)
        && (graphTriggersDropped ?? 0) >= 0
        && (spans?.allSatisfy {
          $0.phase.count <= 120 && $0.name.count <= 120
            && $0.start.isFinite && $0.start >= 0 && $0.ms.isFinite && $0.ms >= 0
        } ?? true)
        && events.allSatisfy { $0.phase.count <= 120 && $0.at.isFinite && $0.at >= 0 }
        && metrics.allSatisfy {
          $0.phase.count <= 120 && $0.name.count <= 120 && $0.value.isFinite && $0.value >= 0
            && $0.unit.count <= 24
        }
    }
  }
  public struct Check: Codable, Sendable {
    public let name: String
    public let status: String
    public let detail: String
    public let ms: Double
  }
  public let version: Int
  public let running: Bool
  public let checkpoint: String?
  public let checks: [Check]
  public let environment: [String: String]
  public let profile: Profile?

  public static func decode(_ body: Any) -> LabReport? {
    guard JSONSerialization.isValidJSONObject(body),
      let data = try? JSONSerialization.data(withJSONObject: body), data.count <= 262_144,
      let report = try? JSONDecoder().decode(Self.self, from: data), report.version == 1,
      report.checks.count <= 128, report.environment.count <= 30, report.profile?.valid != false,
      report.checks.allSatisfy({
        ["pass", "fail", "running", "manual"].contains($0.status)
          && $0.ms.isFinite && $0.ms >= 0 && $0.name.count <= 120 && $0.detail.count <= 4000
      }),
      report.checkpoint == nil || LabCommand(rawValue: report.checkpoint!) != nil
    else { return nil }
    return report
  }
}

@MainActor @Observable
public final class LabSession: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
  public private(set) var status = "Loading bundled renderer…"
  public private(set) var ready = false
  public private(set) var running = false
  public private(set) var report: LabReport?
  public let webView: WKWebView
  private let output: URL
  private let autorun: LabCommand?
  private let automaticSnapshots: Bool
  private let externalGuard: Bool
  private let assets: LabAssets
  private var sequence = 0

  public init(
    assetsRoot: URL, output: URL, autorun: LabCommand?, automaticSnapshots: Bool = true,
    externalGuard: Bool = false
  ) {
    assets = LabAssets(root: assetsRoot)
    self.output = output
    self.autorun = autorun
    self.automaticSnapshots = automaticSnapshots
    self.externalGuard = externalGuard
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    configuration.setURLSchemeHandler(
      LabSchemeHandler(assets: assets.bundled), forURLScheme: "paddock-lab")
    configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
    webView = WKWebView(frame: .zero, configuration: configuration)
    super.init()
    webView.navigationDelegate = self
    webView.isInspectable = true
    let proxy = LabReportProxy()
    proxy.session = self
    webView.configuration.userContentController.add(proxy, name: "labReport")
    webView.load(URLRequest(url: URL(string: LabAssets.origin + "/index.html")!))
  }

  public func command(_ command: LabCommand) {
    guard
      externalGuard
        || ![LabCommand.scalegraphlimit, .scalepdfwalk, .scalegraphrepeat].contains(command)
    else {
      status = "Run limit tests through renderer-lab/profile.mjs with its external memory guard."
      return
    }
    guard ready, !running || command == .theme else { return }
    Task {
      do {
        _ = try await webView.callAsyncJavaScript(
          "return await window.paddockLab.command(command)",
          arguments: ["command": command.rawValue], in: nil, contentWorld: .page)
      } catch { status = "Command failed: \(error.localizedDescription)" }
    }
  }

  public func reload() {
    ready = false
    running = false
    status = "Reloading isolated renderer…"
    webView.reload()
  }

  public func find(_ query: String) {
    guard !query.isEmpty, query.count <= 1000 else { return }
    let configuration = WKFindConfiguration()
    configuration.wraps = true
    webView.find(query, configuration: configuration) { [weak self] result in
      self?.status = result.matchFound ? "Found in rendered text" : "No rendered-text match"
    }
  }

  public func webView(
    _ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
    decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
  ) {
    // No remote navigation, popups or arbitrary attachment access. Blob subframes are
    // reserved for the inert, explicitly sandboxed security fixture.
    let allowed = action.request.url.map { assets.resource(for: $0) != nil } ?? false
    let fixture = action.targetFrame?.isMainFrame == false && action.request.url?.scheme == "blob"
    decisionHandler(allowed || fixture ? .allow : .cancel)
  }

  public func userContentController(
    _ controller: WKUserContentController, didReceive message: WKScriptMessage
  ) {
    guard message.frameInfo.isMainFrame,
      message.frameInfo.request.url?.scheme == "paddock-lab",
      message.frameInfo.request.url?.host == "bundle",
      let value = LabReport.decode(message.body)
    else { return }
    report = value
    running = value.running
    let failures = value.checks.filter { $0.status == "fail" }.count
    let passes = value.checks.filter { $0.status == "pass" }.count
    status =
      value.running
      ? "Running · \(passes) passed · \(failures) failed"
      : "\(passes) passed · \(failures) failed · manual checks remain"
    let first = !ready
    ready = true
    saveReport(value)
    if automaticSnapshots, let checkpoint = value.checkpoint { snapshot(name: checkpoint) }
    if first, let autorun { command(autorun) }
  }

  private func saveReport(_ value: LabReport) {
    do {
      try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(value).write(
        to: output.appendingPathComponent("report.json"), options: .atomic)
    } catch { status = "Could not save report: \(error.localizedDescription)" }
  }

  public func snapshot(name: String = "manual") {
    guard name == "manual" || LabCommand(rawValue: name) != nil else { return }
    sequence += 1
    let file = output.appendingPathComponent(String(format: "%03d-%@.png", sequence, name))
    webView.takeSnapshot(with: nil) { [weak self] image, error in
      guard let self else { return }
      do {
        guard let tiff = image?.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:])
        else { throw error ?? URLError(.cannotDecodeContentData) }
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try png.write(to: file, options: .atomic)
      } catch { self.status = "Snapshot failed: \(error.localizedDescription)" }
    }
  }

  public func webView(
    _ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
    withError error: any Error
  ) {
    status = "Load failed: \(error.localizedDescription)"
  }
  public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
    ready = false
    running = false
    status =
      "WebKit terminated. Reload to rebuild the synthetic fixtures; no user state was loaded."
  }
}
