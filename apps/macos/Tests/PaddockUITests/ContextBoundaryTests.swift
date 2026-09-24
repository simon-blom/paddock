import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite(.serialized)
struct ContextBoundaryTests {
  @Test @MainActor func boundarySurfacesFitBothThemesWithoutAWebView() throws {
    _ = NSApplication.shared
    for json in [
      #"{"before":"m2","title":"Earlier messages summarized","summary":"Keep the original decision and the open question.","working":false,"error":""}"#,
      #"{"title":"","summary":"","working":true,"error":""}"#,
      #"{"before":"m2","title":"Earlier messages are outside the model’s context","summary":"","working":false,"error":"Earlier messages could not be summarized."}"#,
    ] {
      let state = try JSONDecoder().decode(
        StudioState.NativeTranscript.Context.self, from: Data(json.utf8))
      for dark in [false, true] {
        let host = NSHostingView(
          rootView: NativeContextBoundary(context: state)
            .environment(\.colorScheme, dark ? .dark : .light).frame(width: 320))
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width == 320)
        #expect(host.fittingSize.height > 15 && host.fittingSize.height < 200)
        func native(_ view: NSView) -> Bool {
          !String(describing: type(of: view)).contains("WKWebView")
            && view.subviews.allSatisfy(native)
        }
        #expect(native(host))
      }
    }
  }
}
