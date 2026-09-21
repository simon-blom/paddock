import AppKit
import Foundation
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Web-aligned native logs", .timeLimit(.minutes(1))) @MainActor
struct EndpointLogsTests {
  @Test func parsesTheExactSameFixtureAsWebLogView() throws {
    struct Fixture: Decodable {
      let input: String
      let raw: String
      let level: String?
      let effective: String?
      let module: String?
      let message: String?
    }
    var root = URL(fileURLWithPath: #filePath)
    for _ in 0..<5 { root.deleteLastPathComponent() }
    let fixtures = try JSONDecoder().decode(
      [Fixture].self,
      from: Data(contentsOf: root.appending(path: "studio/src/lib/log-lines.fixture.json")))
    var buffer = EndpointLogBuffer()
    buffer.append(fixtures.map(\.input).joined(separator: "\n") + "\n")
    #expect(buffer.lines.count == fixtures.count)
    for (line, fixture) in zip(buffer.lines, fixtures) {
      #expect(
        line.raw == fixture.raw && line.level == fixture.level
          && line.effective == fixture.effective)
      #expect(line.module == fixture.module && line.message == fixture.message)
    }
    #expect(buffer.visible(query: "", minimum: "WARN").count == 4)
    #expect(buffer.visible(query: " DETAILs ", minimum: "WARN").count == 1)
  }
  @Test func retainedHistoryAndRenderedRowsStayBounded() {
    var buffer = EndpointLogBuffer()
    buffer.append(
      (0..<5000).map { "2026-09-16T09:12:04Z INFO runner: event \($0)" }.joined(separator: "\n")
        + "\n")
    #expect(buffer.lines.count == 4000 && buffer.lines.first?.message == "event 1000")
    #expect(buffer.visible(query: "", minimum: "all").count == 1500)
    #expect(buffer.visible(query: "event 1000", minimum: "all").count == 1)
    buffer.append(String(repeating: String(repeating: "x", count: 4096) + "\n", count: 300))
    #expect(buffer.bytes <= 1024 * 1024 && buffer.lines.count < 300)
  }
  @Test func appendAndEvictionPreserveRetainedSelection() throws {
    let view = LogScrollView(frame: NSRect(x: 0, y: 0, width: 600, height: 250))
    var buffer = EndpointLogBuffer()
    buffer.append("first\nsecond\nthird\n")
    view.update(buffer.lines, jump: 0)
    let range = (view.text.string as NSString).range(of: "second\nthird")
    view.text.setSelectedRange(range)
    buffer.append("fourth\n")
    view.update(buffer.lines, jump: 0)
    #expect(view.text.selectedRange() == range)
    view.update(Array(buffer.lines.dropFirst()), jump: 0)
    #expect(
      (view.text.string as NSString).substring(with: view.text.selectedRange()) == "second\nthird")
    #expect(view.text.string == "second\nthird\nfourth\n")
  }
  @Test func scrolledBackLogsDoNotJumpUntilLatest() {
    let view = LogScrollView(frame: NSRect(x: 0, y: 0, width: 600, height: 180))
    var buffer = EndpointLogBuffer()
    buffer.append((0..<100).map { "event \($0)\n" }.joined())
    view.update(buffer.lines, jump: 0)
    view.contentView.scroll(to: .zero)
    NotificationCenter.default.post(
      name: NSView.boundsDidChangeNotification, object: view.contentView)
    #expect(!view.following)
    buffer.append("latest\n")
    view.update(buffer.lines, jump: 0)
    #expect(view.contentView.bounds.origin.y == 0 && !view.following)
    view.update(buffer.lines, jump: 1)
    #expect(view.following && view.contentView.bounds.origin.y > 0)
  }
  @Test func cancellingViewClosesOnlyItsReadSubscription() async throws {
    let client = LogFixture()
    let model = EndpointLogsModel(client: client)
    let task = Task { await model.follow(port: 13495) }
    for _ in 0..<100 {
      if await client.polls > 0 { break }
      try await Task.sleep(for: .milliseconds(10))
    }
    task.cancel()
    await task.value
    #expect(await client.closed == ["fixture-13495"])
    #expect(model.state == "paused")
  }
  @Test func incrementalMountCostWith1500VisibleLines() {
    var buffer = EndpointLogBuffer()
    buffer.append((0..<1500).map { "2026-09-16T09:12:04Z INFO runner: event \($0)\n" }.joined())
    let view = LogScrollView(frame: NSRect(x: 0, y: 0, width: 700, height: 320))
    let initial = CFAbsoluteTimeGetCurrent()
    view.update(buffer.lines, jump: 0)
    let cold = (CFAbsoluteTimeGetCurrent() - initial) * 1000
    var samples: [Double] = []
    for i in 1500..<1600 {
      buffer.append("2026-09-16T09:12:04Z INFO runner: event \(i)\n")
      let lines = buffer.visible(query: "", minimum: "all")
      let start = CFAbsoluteTimeGetCurrent()
      view.update(lines, jump: 0)
      samples.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }
    #expect(view.text.string.components(separatedBy: "\n").count == 1501)
    samples.sort()
    print(
      "NATIVE_LOG_PERF cold_ms=\(cold) update_p99_ms=\(samples[98]) update_max_ms=\(samples[99])")
  }
  @Test func lightAndDarkLogSurfacesRenderOffscreen() async throws {
    for dark in [false, true] {
      let model = EndpointLogsModel(client: LogFixture())
      let task = Task { await model.follow(port: 13495) }
      for _ in 0..<100 {
        if !model.buffer.lines.isEmpty { break }
        try await Task.sleep(for: .milliseconds(10))
      }
      task.cancel()
      await task.value
      let host = NSHostingView(
        rootView: EndpointLogsView(model: model, port: 13495)
          .padding(24).frame(width: 640).background(PaddockStyle.canvas).environment(
            \.colorScheme, dark ? .dark : .light
          ).environment(\.scenePhase, .inactive))
      host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      let size = host.fittingSize
      #expect(size.width == 640 && size.height > 350 && size.height < 700)
      host.frame = NSRect(origin: .zero, size: size)
      host.layoutSubtreeIfNeeded()
      if let directory = ProcessInfo.processInfo.environment["PADDOCK_LOG_SNAPSHOTS"],
        let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)
      {
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let folder = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try bitmap.representation(using: .png, properties: [:])?.write(
          to: folder.appending(path: "logs-\(dark ? "dark" : "light").png"))
      }
    }
  }
}

private actor LogFixture: ManagerLoading {
  private(set) var polls = 0
  private(set) var closed: [String] = []
  func snapshot() async throws -> ManagerSnapshot { try endpointFixture() }
  func logs(_ command: LogCommand) async throws -> LogReply {
    let value: [String: Any]
    switch command {
    case .open(let port): value = ["id": "fixture-\(port)", "state": "connecting", "text": ""]
    case .poll(let id):
      polls += 1
      value = ["id": id, "state": "live", "text": "2026-09-16T09:12:04Z INFO runner: healthy\n"]
    case .close(let id):
      closed.append(id)
      value = ["state": "closed", "text": ""]
    }
    return try JSONDecoder().decode(
      LogReply.self, from: JSONSerialization.data(withJSONObject: value))
  }
}
