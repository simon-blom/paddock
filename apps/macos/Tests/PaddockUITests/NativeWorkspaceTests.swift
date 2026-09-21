import AppKit
import Foundation
import PaddockClient
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native content workspace boundary") @MainActor
struct NativeWorkspaceTests {
  @Test func composerCornerClearanceIsHorizontalOnly() {
    #expect(StudioComposerLayout.horizontalControlInset == 16)
    #expect(StudioComposerLayout.bottomInset == 8)
    #expect(StudioComposerLayout.contentInset == 8)
  }
  @Test func composerFollowsContentBoundsWithoutASecondInsetOrCap() throws {
    func viewport(left: Double, width: Double) throws -> StudioState.Viewport {
      try JSONDecoder().decode(
        StudioState.Viewport.self,
        from: JSONSerialization.data(withJSONObject: ["left": left, "width": width]))
    }
    for (available, left, width): (CGFloat, Double, Double) in [
      (1200, 220, 760), (650, 28, 594), (1200, 628, 544),
      (1400, 80, 1240), (320, 28, 264), (1200, 212.5, 760),
    ] {
      let rect = StudioColumnLayout.resolve(
        available: available, viewport: try viewport(left: left, width: width))
      #expect(Double(rect.minX) == left)
      #expect(Double(rect.width) == width)
    }
    #expect(StudioColumnLayout.resolve(available: 1200, viewport: nil).width == 760)
    #expect(StudioColumnLayout.resolve(available: 320, viewport: nil).width == 264)
    let stale = StudioColumnLayout.resolve(
      available: 500, viewport: try viewport(left: 450, width: 760))
    #expect(stale.maxX == 500)
  }
  @Test func editorHeightAndSelectionStayBounded() {
    #expect(StudioDraftEditor.composerHeight(0) == 72)
    #expect(StudioDraftEditor.composerHeight(122.2) == 123)
    #expect(StudioDraftEditor.composerHeight(10000) == 264)
    #expect(
      StudioDraftEditor.clampedSelection(NSRange(location: 2, length: 5), length: 20)
        == NSRange(location: 2, length: 5))
    #expect(
      StudioDraftEditor.clampedSelection(NSRange(location: 9, length: 5), length: 10)
        == NSRange(location: 9, length: 1))
    #expect(
      StudioDraftEditor.clampedSelection(NSRange(location: 19, length: 5), length: 3)
        == NSRange(location: 3, length: 0))
  }

  @Test func editorUsesTextKitTwoAndGrowsThenScrolls() async throws {
    _ = NSApplication.shared
    var height: CGFloat = 0
    let controller = NSHostingController(
      rootView:
        StudioDraftEditor(
          text: .constant(String(repeating: "Native text layout\n", count: 30)), onSend: {},
          onFiles: { _ in }, onHeight: { height = $0 }
        )
        .frame(width: 320, height: 264))
    controller.view.frame = NSRect(x: 0, y: 0, width: 320, height: 264)
    controller.view.layoutSubtreeIfNeeded()
    func editor(in view: NSView) -> DraftTextView? {
      if let value = view as? DraftTextView { return value }
      return view.subviews.lazy.compactMap { editor(in: $0) }.first
    }
    let text = try #require(editor(in: controller.view))
    #expect(text.textLayoutManager != nil)
    #expect(text.frame.width <= text.enclosingScrollView!.contentView.bounds.width + 1)
    #expect(text.allowsUndo)
    let disabledHost = NSHostingController(
      rootView: StudioDraftEditor(
        text: .constant("Read only during transfer"), onSend: {}, onFiles: { _ in }
      ).frame(width: 320, height: 72).disabled(true))
    disabledHost.view.frame = NSRect(x: 0, y: 0, width: 320, height: 72)
    disabledHost.view.layoutSubtreeIfNeeded()
    let disabledEditor = try #require(editor(in: disabledHost.view))
    #expect(!disabledEditor.isEditable)
    #expect(disabledEditor.onSend == nil && disabledEditor.onFiles == nil)
    text.measureHeight()
    try await Task.sleep(for: .milliseconds(80))
    #expect(height == 264)
    #expect(text.enclosingScrollView?.hasVerticalScroller == true)
    text.string = "Short"
    text.measureHeight()
    try await Task.sleep(for: .milliseconds(80))
    #expect(height == 72)
  }
  @Test func bootstrapAcceptsOnlyExactPrivateLoopbackOrigins() throws {
    func host(_ origin: String, token: String = String(repeating: "a", count: 64)) throws
      -> StudioHost
    {
      let data = try JSONSerialization.data(withJSONObject: [
        "origin": origin, "cookieName": "paddock_desktop_session", "session": token,
      ])
      return try JSONDecoder().decode(StudioHost.self, from: data)
    }
    #expect(StudioWorkspace.validHost(try host("http://127.0.0.1:12345")))
    for origin in [
      "https://example.com", "http://localhost:12345", "http://127.0.0.1",
      "http://127.0.0.1:12345/path", "http://127.0.0.1:12345/?key=secret",
      "http://user@127.0.0.1:12345", "http://127.0.0.1:12345/#fragment",
    ] {
      #expect(!StudioWorkspace.validHost(try host(origin)))
    }
    #expect(!StudioWorkspace.validHost(try host("http://127.0.0.1:12345", token: "short")))
  }

  @Test func commandArgumentsPreserveTextAsData() throws {
    let value: StudioValue = .object([
      "text": .string("Quotes \" ' ` ${text} 🐕\n<script>"), "switch": .bool(false),
      "pages": .array([.number(1), .null]),
    ])
    #expect(try JSONDecoder().decode(StudioValue.self, from: JSONEncoder().encode(value)) == value)
  }

  @Test func composerDraftBelongsToAppAndRequiresExplicitQuit() async {
    let model = WorkspaceModel(client: UnavailableWorkspaceFixture())
    #expect(!model.studioNeedsQuitConfirmation)
    model.draft = StudioDraft(message: "Retain this native draft")
    await model.refresh()
    #expect(model.draft.message == "Retain this native draft")
    #expect(model.studioNeedsQuitConfirmation)
    model.draft = StudioDraft()
    #expect(!model.studioNeedsQuitConfirmation)
    await model.shutdown()
  }
}

private struct UnavailableWorkspaceFixture: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    throw ManagerError.core("Synthetic offline manager")
  }
  func close() async {}
}
