import Foundation
import PaddockRendererHost
import WebKit

/// Opt-in, lab-only Web Inspector diagnostic. Inspector changes execution and
/// memory behaviour: no sample from this run is a performance acceptance bar.
@MainActor
enum LayerCapture {
  static func run(session: LabSession, output: URL, command: LabCommand) async {
    var result: [String: Any] = [
      "diagnosticOnly": true, "started": Date().timeIntervalSince1970 * 1000,
    ]
    var rows: [[String: Any]] = []
    var capturedBytes = 0
    do {
      for _ in 0..<100 {
        if session.ready { break }
        try await Task.sleep(for: .milliseconds(100))
      }
      guard session.ready else { throw failure("Lab did not become ready") }
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
          (try? await frontend.evaluateJavaScript("!!globalThis.WI?.mainTarget?.LayerTreeAgent"))
            as? Bool == true
        {
          break
        }
        try await Task.sleep(for: .milliseconds(250))
      }
      guard let frontend else { throw failure("Inspector frontend did not connect") }
      // Inspector's manager already enables LayerTree. Do not enable twice.
      let script = """
        const target = WI.mainTarget;
        const call = (agent, method, ...args) => new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error(`${method} timed out`)), 5000);
          agent[method](...args, (error, data) => {
            clearTimeout(timer); error ? reject(new Error(error)) : resolve(data);
          });
        });
        const root = await new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('DOM root timed out')), 5000);
          WI.domManager.requestDocument(root => { clearTimeout(timer); resolve(root); });
        });
        if (!root) throw new Error('Missing main-page DOM root');
        const layers = await call(target.LayerTreeAgent, 'layersForNode', root.id);
        if (layers.length > 1024) throw new Error('Layer diagnostic limit exceeded');
        return JSON.stringify(await Promise.all(layers.map(async layer => {
          const reasons = await call(target.LayerTreeAgent, 'reasonsForCompositingLayer', layer.layerId).catch(error => ({error: error.message}));
          const node = WI.domManager.nodeForId(layer.nodeId);
          return {...layer, reasons, nodeName: node?.nodeName(), attributes: node?.attributes().map(({name, value}) => ({name, value}))};
        })));
        """
      session.command(command)
      let deadline = Date().addingTimeInterval(180)
      while Date() < deadline {
        let value = try await frontend.callAsyncJavaScript(
          script, arguments: [:], in: nil, contentWorld: .page)
        guard let json = value as? String, json.utf8.count < 2 * 1024 * 1024,
          let data = json.data(using: .utf8)
        else { throw failure("Missing or excessive layer data") }
        capturedBytes += data.count
        guard capturedBytes <= 32 * 1024 * 1024 else {
          throw failure("Layer capture budget exceeded")
        }
        let layers = try JSONSerialization.jsonObject(with: data)
        rows.append([
          "at": Date().timeIntervalSince1970 * 1000,
          "phase": session.report?.profile?.phase ?? "starting", "layers": layers,
        ])
        if session.report?.profile?.phase == "complete" { break }
        try await Task.sleep(for: .milliseconds(200))
      }
      guard session.report?.profile?.phase == "complete" else {
        throw failure("Layer workload deadline exceeded")
      }
      result["status"] = "completed"
    } catch {
      result["status"] = "failed"
      result["error"] = error.localizedDescription
      result["detail"] = String(describing: (error as NSError).userInfo)
    }
    result["ended"] = Date().timeIntervalSince1970 * 1000
    result["samples"] = rows
    if let data = try? JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]) {
      try? data.write(
        to: output.appendingPathComponent("layer-capture.json"), options: .withoutOverwriting)
    }
  }

  private static func failure(_ message: String) -> NSError {
    NSError(
      domain: "PaddockRenderingLab.LayerCapture", code: 1,
      userInfo: [NSLocalizedDescriptionKey: message])
  }
}
