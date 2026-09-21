import Foundation
import PaddockRendererHost
import WebKit

/// Developer-only executable, never PaddockMac or the shared renderer host.
/// Web Inspector's snapshot performs GC. Attach only after the unforced 45s
/// plateau; all subsequent samples are diagnostics, never a memory/latency bar.
/// Private selectors are feature-tested and confined to this opt-in lab tool.
@MainActor
enum HeapCapture {
  static func run(session: LabSession, output: URL) async {
    let deadline = Date().addingTimeInterval(300)
    while session.report?.profile?.phase != "complete", Date() < deadline {
      try? await Task.sleep(for: .milliseconds(500))
    }
    guard session.report?.profile?.phase == "complete" else { return }
    try? await Task.sleep(for: .seconds(60))
    let started = Date().timeIntervalSince1970 * 1000
    var result: [String: Any] = [
      "started": started, "snapshotForcesGC": true, "snapshotRequested": false,
      "diagnosticOnly": true, "scope": "isolated Rendering Lab main-page JS heap",
    ]
    do {
      result["beforeSnapshot"] = try await session.webView.evaluateJavaScript(
        "window.paddockLab.memoryStats?.() ?? null")
      let preferences = session.webView.configuration.preferences
      guard preferences.responds(to: NSSelectorFromString("_setDeveloperExtrasEnabled:")) else {
        throw failure("Installed WebKit has no local inspector preference")
      }
      let oldExtras = preferences.value(forKey: "developerExtrasEnabled")
      preferences.setValue(true, forKey: "developerExtrasEnabled")
      defer { preferences.setValue(oldExtras, forKey: "developerExtrasEnabled") }
      guard session.webView.responds(to: NSSelectorFromString("_inspector")),
        let inspector = session.webView.value(forKey: "_inspector") as? NSObject,
        inspector.responds(to: NSSelectorFromString("connect")),
        inspector.responds(to: NSSelectorFromString("inspectorWebView"))
      else { throw failure("Installed WebKit has no supported inspector diagnostic interface") }
      inspector.perform(NSSelectorFromString("connect"))
      defer {
        if inspector.responds(to: NSSelectorFromString("close")) {
          inspector.perform(NSSelectorFromString("close"))
        }
      }
      var frontend: WKWebView?
      for _ in 0..<40 {
        frontend = inspector.value(forKey: "inspectorWebView") as? WKWebView
        if let frontend,
          (try? await frontend.evaluateJavaScript("!!globalThis.WI?.mainTarget?.HeapAgent"))
            as? Bool == true
        {
          break
        }
        try await Task.sleep(for: .milliseconds(250))
      }
      guard let frontend else { throw failure("Inspector frontend did not connect") }
      result["frontendURL"] = frontend.url?.absoluteString
      result["snapshotRequested"] = true
      let script = """
        const target = globalThis.WI?.mainTarget;
        if (!target?.HeapAgent) throw new Error('Main-page Heap agent unavailable');
        // The Inspector may have enabled Heap during its own initialization.
        // Its manager owns that state; enabling the protocol twice is an error.
        WI.heapManager.enable();
        return await new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('Heap snapshot timed out')), 20000);
          target.HeapAgent.snapshot((error, timestamp, data) => {
            clearTimeout(timer);
            if (error) reject(new Error(error)); else resolve(data);
          });
        });
        """
      var snapshots: [[String: Any]] = []
      for name in ["post-idle", "post-idle-2"] {
        if !snapshots.isEmpty { try await Task.sleep(for: .seconds(1)) }
        let snapshot = try await frontend.callAsyncJavaScript(
          script, arguments: [:], in: nil, contentWorld: .page)
        guard let snapshot = snapshot as? String, snapshot.utf8.count < 96 * 1024 * 1024 else {
          throw failure("Missing or excessive heap snapshot")
        }
        let data = Data(snapshot.utf8)
        let file = "\(name).heapsnapshot.json"
        try data.write(to: output.appendingPathComponent(file), options: .withoutOverwriting)
        snapshots.append([
          "file": file, "bytes": data.count, "at": Date().timeIntervalSince1970 * 1000,
        ])
      }
      result["snapshots"] = snapshots
      result["status"] = "completed"
    } catch {
      result["status"] = "failed"
      result["error"] = error.localizedDescription
      result["detail"] = String(describing: (error as NSError).userInfo)
    }
    result["ended"] = Date().timeIntervalSince1970 * 1000
    if let data = try? JSONSerialization.data(
      withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    {
      try? data.write(
        to: output.appendingPathComponent("heap-capture.json"), options: .withoutOverwriting)
    }
  }

  private static func failure(_ message: String) -> NSError {
    NSError(
      domain: "PaddockRenderingLab.HeapCapture", code: 1,
      userInfo: [NSLocalizedDescriptionKey: message])
  }
}
