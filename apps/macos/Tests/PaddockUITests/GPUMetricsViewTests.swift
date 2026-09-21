import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("GPU metrics presentation", .serialized) @MainActor
struct GPUMetricsViewTests {
  @Test func boundedNativePanelFitsBothThemesWithoutOpeningWindows() throws {
    for count in [0, 1, 8] {
      let rows = (0..<count).map { index in
        "{\"port\":\(11540 + index),\"pid\":\(42 + index),\"metal\":{\"allocated_bytes\":12000000000,\"completed_commands\":8,\"gpu_seconds_total\":0.3,\"last_command_ms\":20}}"
      }.joined(separator: ",")
      let data = Data(
        """
        {"available":true,"ts":100,"gpus":[{"index":0,"name":"Apple M5 Max",
        "metal":{"unified_memory_total":137438953472,"recommended_working_set":103079215104}}],
        "reconciliation":{"ts":100,"runners":[\(rows)]}}
        """.utf8)
      let snapshot = try ManagerWire.decode(GPUSnapshot.self, from: data)
      var history = GPUHistory()
      history.ingest(snapshot, at: Date(timeIntervalSince1970: 100))
      for dark in [false, true] {
        let host = NSHostingView(
          rootView: GPUMetricsView(snapshot: snapshot, runners: [], history: history)
            .environment(\.colorScheme, dark ? .dark : .light))
        host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        host.frame = NSRect(x: 0, y: 0, width: 340, height: 540)
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width == 340)
        #expect(host.fittingSize.height <= 540)
      }
    }
  }
}
