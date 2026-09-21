import Foundation
import Testing

@testable import PaddockRendererHost

@Suite("Renderer resource and report boundary")
struct RendererTests {
  @Test func pdfReopenPlanIsAvailableToAutorun() {
    #expect(LabCommand(rawValue: "scalepdfrepeat") == .scalepdfrepeat)
  }

  @Test func mixedGraphReopenPlanIsAvailableToAutorun() {
    #expect(LabCommand(rawValue: "scalegraphrepeat") == .scalegraphrepeat)
  }

  @Test func resourceAllowlist() throws {
    let assets = LabAssets(root: URL(fileURLWithPath: "/tmp/render-assets"))
    #expect(
      assets.resource(for: URL(string: "paddock-lab://bundle/assets/a.wasm")!)?.path
        == "/tmp/render-assets/assets/a.wasm")
    for address in [
      "https://bundle/assets/a.wasm", "file:///etc/passwd", "paddock-lab://other/index.html",
      "paddock-lab://bundle/%2e%2e/secret.json", "paddock-lab://bundle/secret.key",
      "paddock-lab://user@bundle/index.html", "paddock-lab://bundle:80/index.html",
      "paddock-lab://bundle/%00.js", "paddock-lab://bundle/foo%5csecret.js",
    ] {
      #expect(assets.resource(for: try #require(URL(string: address))) == nil, "\(address)")
    }
  }

  @Test func diagnosticMessagesCannotChoosePaths() {
    let base: [String: Any] = ["version": 1, "running": false, "checks": [], "environment": [:]]
    #expect(LabReport.decode(base) != nil)
    var invalid = base
    invalid["checkpoint"] = "../../outside"
    #expect(LabReport.decode(invalid) == nil)
    invalid = base
    invalid["version"] = 2
    #expect(LabReport.decode(invalid) == nil)
    invalid = base
    invalid["checks"] = [["name": "x", "status": "pass", "detail": "", "ms": -1]]
    #expect(LabReport.decode(invalid) == nil)
  }

  @Test func symlinksCannotEscapeTheBundle() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(
      "renderer-path-test-\(UUID().uuidString)")
    let bundle = folder.appendingPathComponent("bundle")
    try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    try FileManager.default.createSymbolicLink(
      at: bundle.appendingPathComponent("escape"), withDestinationURL: folder)
    let assets = LabAssets(root: bundle)
    #expect(assets.resource(for: URL(string: "paddock-lab://bundle/escape/private.json")!) == nil)
    #expect(
      assets.resource(for: URL(string: "paddock-lab://bundle/index.html")!)
        == bundle.appendingPathComponent("index.html"))
  }

  @Test func bridgeIsBoundedAndRejectsUnknownStatuses() {
    let base: [String: Any] = ["version": 1, "running": false, "checks": [], "environment": [:]]
    var payload = base
    payload["checks"] = [["name": "x", "status": "execute", "detail": "", "ms": 0]]
    #expect(LabReport.decode(payload) == nil)
    payload["checks"] = Array(
      repeating: ["name": "x", "status": "pass", "detail": "", "ms": 0], count: 129)
    #expect(LabReport.decode(payload) == nil)
    payload = base
    payload["environment"] = ["large": String(repeating: "x", count: 262_144)]
    #expect(LabReport.decode(payload) == nil)
    payload = base
    payload["checks"] = [["name": "x", "status": "pass", "detail": "", "ms": Double.infinity]]
    #expect(LabReport.decode(payload) == nil)
  }

  @Test func profileDiagnosticsAreBounded() {
    var payload: [String: Any] = ["version": 1, "running": false, "checks": [], "environment": [:]]
    payload["profile"] = [
      "phase": "pdf/open", "events": [["phase": "pdf/open", "at": 1000]],
      "metrics": [["phase": "pdf/open", "name": "elapsed", "value": 50, "unit": "ms"]],
    ]
    #expect(LabReport.decode(payload)?.profile?.metrics.count == 1)
    payload["profile"] = [
      "phase": "x", "events": [], "metrics": [],
      "spans": [["phase": "x", "name": "mount", "start": 10, "ms": 3]],
    ]
    #expect(LabReport.decode(payload)?.profile?.spans?.count == 1)
    payload["profile"] = [
      "phase": "x", "events": [], "metrics": [],
      "spans": Array(repeating: ["phase": "x", "name": "mount", "start": 0, "ms": 3], count: 513),
    ]
    #expect(LabReport.decode(payload) == nil)
    payload["profile"] = [
      "phase": "x", "events": [],
      "metrics":
        [["phase": "x", "name": "bytes", "value": -1, "unit": "bytes"]],
    ]
    #expect(LabReport.decode(payload) == nil)
    payload["profile"] = [
      "phase": "x", "events": Array(repeating: ["phase": "x", "at": 0], count: 257), "metrics": [],
    ]
    #expect(LabReport.decode(payload) == nil)
    payload["profile"] = [
      "phase": "x", "events": [],
      "metrics": Array(
        repeating: ["phase": "x", "name": "m", "value": 0, "unit": "ms"], count: 1025),
    ]
    #expect(LabReport.decode(payload) == nil)
  }

  @Test func graphTriggerDiagnosticsSurviveTheBridgeAndStayBounded() throws {
    let row: [String: Any] = [
      "phase": "graph/open", "kind": "process", "start": 10,
      "ms": 9, "causes": ["graph:eachNodeAttributesUpdated"],
    ]
    func decode(_ rows: [[String: Any]], dropped: Int = 0) -> LabReport? {
      LabReport.decode([
        "version": 1, "running": false, "checks": [], "environment": [:],
        "profile": [
          "phase": "graph/open", "events": [], "metrics": [],
          "graphTriggers": rows, "graphTriggersDropped": dropped,
        ],
      ])
    }
    let report = try #require(decode([row]))
    #expect(report.profile?.graphTriggers?.first?.causes == ["graph:eachNodeAttributesUpdated"])
    let roundtrip = try JSONDecoder().decode(LabReport.self, from: JSONEncoder().encode(report))
    #expect(roundtrip.profile?.graphTriggers?.count == 1)
    #expect(decode(Array(repeating: row, count: 769)) == nil)
    #expect(decode([row], dropped: -1) == nil)
    for (key, value) in [
      ("ms", -1 as Any), ("kind", "execute" as Any),
      ("nodes", 200_001 as Any), ("edges", -1 as Any),
      ("causes", Array(repeating: "x", count: 17) as Any),
    ] {
      var invalid = row
      invalid[key] = value
      #expect(decode([invalid]) == nil)
    }
  }
}
