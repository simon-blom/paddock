import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Inline native dictation parity", .serialized) @MainActor
struct DictationDecorationTests {
  struct Fixture: Decodable {
    let name: String, draft: String, provisional: String, ghost: String, committed: String
  }
  @Test func sharedWebFixturesMatchGhostSpacingAndFinalization() throws {
    let url = try #require(
      Bundle.module.url(forResource: "dictation", withExtension: "json", subdirectory: "Fixtures"))
    for row in try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url)) {
      let (_, editor) = try editor(row.draft)
      editor.provisionalDictation = row.provisional
      #expect(
        DraftDictationOverlay.suffix(after: row.draft, provisional: row.provisional) == row.ghost,
        "\(row.name)")
      #expect(editor.string == row.draft)
      editor.provisionalDictation = ""
      #expect(editor.appendDictated(row.provisional))
      #expect(editor.string == row.committed, "\(row.name)")
    }
  }

  @Test func ghostCannotEnterSendCopySelectionAccessibilityValueOrUndo() async throws {
    var draft = "Edit this sentence."
    var sent = ""
    let host = NSHostingController(
      rootView: StudioDraftEditor(
        text: Binding(get: { draft }, set: { draft = $0 }), onSend: { sent = draft },
        provisionalDictation: "Heard but not confirmed."))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 360, height: 72),
      styleMask: [.titled], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    defer { window.close() }
    host.view.frame = NSRect(x: 0, y: 0, width: 360, height: 72)
    host.view.layoutSubtreeIfNeeded()
    let editor = try #require(find(host.view))
    window.makeFirstResponder(editor)
    let storage = try #require(editor.textStorage)
    let undo = try #require(editor.undoManager)
    let range = NSRange(location: 2, length: 8)
    editor.setSelectedRange(range)
    let overlay = try #require(editor.dictationOverlay)
    for provisional in ["Heard", "Heard this instead.", "Heard this instead. "] {
      editor.provisionalDictation = provisional
      #expect(editor.dictationOverlay === overlay)
      #expect(editor.textStorage === storage)
      #expect(editor.selectedRange() == range)
      #expect(!undo.canUndo)
      #expect(editor.accessibilityValue() == draft)
      #expect(overlay.hitTest(.zero) == nil)
      #expect(!overlay.isAccessibilityElement())
    }
    editor.onSend?()
    #expect(sent == "Edit this sentence.")
    editor.setSelectedRange(NSRange(location: 0, length: (draft as NSString).length))
    #expect(editor.selectedRange().length == (draft as NSString).length)
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    #expect(editor.writeSelection(to: pasteboard, types: editor.writablePasteboardTypes))
    #expect(pasteboard.string(forType: .string) == "Edit this sentence.")
    editor.setSelectedRange(range)
    editor.provisionalDictation = ""
    #expect(editor.dictationOverlay == nil)
    #expect(editor.appendDictated("Heard this instead."))
    #expect(draft == "Edit this sentence. Heard this instead.")
    #expect(editor.selectedRange() == range)
    undo.undo()
    #expect(draft == "Edit this sentence.")
    undo.redo()
    #expect(draft == "Edit this sentence. Heard this instead.")
    try await Task.sleep(for: .milliseconds(20))
  }

  @Test func revisionsTrackTypingWithoutTakingOverIME() async throws {
    let (host, editor) = try editor("Hej")
    defer { withExtendedLifetime(host) {} }
    editor.provisionalDictation = "världen"
    editor.insertText(" ", replacementRange: NSRange(location: 3, length: 0))
    #expect(editor.dictationOverlay?.suffix == "världen")
    editor.setMarkedText(
      "編集中", selectedRange: NSRange(location: 1, length: 0),
      replacementRange: NSRange(location: 0, length: 0))
    let marked = editor.markedRange()
    editor.provisionalDictation = "världen igen."
    #expect(editor.hasMarkedText() && editor.markedRange() == marked)
    #expect(!editor.appendDictated("Wait for IME."))
    editor.unmarkText()
    #expect(editor.appendDictated("Now final."))
    #expect(editor.string.hasSuffix("Now final."))
    #expect(editor.textLayoutManager != nil)
    try await Task.sleep(for: .milliseconds(20))
  }

  @Test func queuedFinalsResumeAfterIMEAndAreAcknowledgedOnlyOnce() async throws {
    let (host, editor) = try editor("Original")
    defer { withExtendedLifetime(host) {} }
    let coordinator = try #require(editor.delegate as? StudioDraftEditor.Coordinator)
    var acknowledgements: [Int] = []
    coordinator.parent.onDictated = { _, index in acknowledgements.append(index) }
    coordinator.parent.dictationSession = "first"
    coordinator.parent.dictation = try JSONDecoder().decode(
      [StudioState.Audio.Item].self,
      from: Data(#"[{"index":0,"text":"Final sentence."}]"#.utf8))
    editor.setMarkedText(
      "あ", selectedRange: NSRange(location: 1, length: 0),
      replacementRange: NSRange(location: 0, length: 0))
    coordinator.scheduleDictation()
    try await Task.sleep(for: .milliseconds(20))
    #expect(acknowledgements.isEmpty && !editor.string.contains("Final sentence."))
    editor.unmarkText()
    try await Task.sleep(for: .milliseconds(20))
    #expect(acknowledgements == [0])
    #expect(editor.string == "あOriginal Final sentence.")
    coordinator.scheduleDictation()
    try await Task.sleep(for: .milliseconds(20))
    #expect(acknowledgements == [0] && editor.string == "あOriginal Final sentence.")
  }

  @Test func updatesPreserveScrolledEditingAndReleaseLayout() async throws {
    let draft = String(repeating: "An earlier paragraph that remains editable.\n", count: 200)
    let (host, editor) = try editor(draft)
    defer { withExtendedLifetime(host) {} }
    let scroll = try #require(editor.enclosingScrollView)
    editor.provisionalDictation = "Still provisional."
    editor.setSelectedRange(NSRange(location: 3, length: 4))
    scroll.contentView.scroll(to: NSPoint(x: 0, y: 120))
    let origin = scroll.contentView.bounds.origin
    weak let released = editor.dictationOverlay
    var times: [Double] = []
    for index in 0..<100 {
      let started = ContinuousClock.now
      editor.provisionalDictation = "Still provisional, revision \(index)."
      let duration = started.duration(to: .now).components
      times.append(Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15)
      #expect(editor.selectedRange() == NSRange(location: 3, length: 4))
      #expect(scroll.contentView.bounds.origin == origin)
    }
    editor.provisionalDictation = ""
    #expect(editor.appendDictated("Confirmed."))
    #expect(scroll.contentView.bounds.origin == origin)
    try await Task.sleep(for: .milliseconds(20))
    #expect(released == nil)
    if ProcessInfo.processInfo.environment["PADDOCK_DICTATION_CAPTURE"] != nil {
      print(
        "Dictation update ms: p50=\(times.sorted()[50]), p99=\(times.sorted()[98]) (8.6k-character draft, 100 revisions)"
      )
    }
  }

  @Test func inlineDecorationPaintsInBothAppearances() async throws {
    for appearance in [NSAppearance.Name.aqua, .darkAqua] {
      let (host, editor) = try editor("A short draft.")
      defer { withExtendedLifetime(host) {} }
      editor.appearance = NSAppearance(named: appearance)
      editor.drawsBackground = true
      editor.backgroundColor = .windowBackgroundColor
      editor.provisionalDictation = "These words are provisional. They wrap onto the next line."
      try await Task.sleep(for: .milliseconds(20))
      let overlay = try #require(editor.dictationOverlay)
      let image = try #require(overlay.bitmapImageRepForCachingDisplay(in: overlay.bounds))
      overlay.cacheDisplay(in: overlay.bounds, to: image)
      var painted = 0
      for y in 0..<image.pixelsHigh {
        for x in 0..<image.pixelsWide where (image.colorAt(x: x, y: y)?.alphaComponent ?? 0) > 0.1 {
          painted += 1
        }
      }
      #expect(painted > 100, "The ghost must actually draw, not only report a height")
      if let directory = ProcessInfo.processInfo.environment["PADDOCK_DICTATION_CAPTURE"] {
        let combined = try #require(editor.bitmapImageRepForCachingDisplay(in: editor.bounds))
        editor.cacheDisplay(in: editor.bounds, to: combined)
        try combined.representation(using: .png, properties: [:])?.write(
          to:
            URL(fileURLWithPath: directory).appendingPathComponent(
              "inline-\(appearance.rawValue).png"))
      }
    }
  }

  @Test func wrappingHeightAndReclamationRemainNativeInBothAppearances() async throws {
    for appearance in [NSAppearance.Name.aqua, .darkAqua] {
      let (host, editor) = try editor("A short draft.")
      defer { withExtendedLifetime(host) {} }
      editor.appearance = NSAppearance(named: appearance)
      editor.provisionalDictation = String(
        repeating: "A provisional sentence with wrapping. ", count: 20)
      var heights: [CGFloat] = []
      editor.onHeight = { heights.append($0) }
      editor.measureHeight()
      try await Task.sleep(for: .milliseconds(30))
      let overlay = try #require(editor.dictationOverlay)
      #expect(overlay.contentHeight > 264)
      #expect(editor.frame.height >= overlay.contentHeight)
      #expect(heights.last == 264)
      let wide = overlay.contentHeight
      editor.setFrameSize(NSSize(width: 160, height: editor.frame.height))
      #expect(overlay.contentHeight > wide)
      #expect(editor.textLayoutManager != nil && overlay.layout.textContainer != nil)
      // Render offscreen. No visible fixture windows, no microphone permission.
      let image = try #require(overlay.bitmapImageRepForCachingDisplay(in: overlay.bounds))
      overlay.cacheDisplay(in: overlay.bounds, to: image)
      #expect(image.pixelsWide > 0 && image.pixelsHigh > 0)
      editor.provisionalDictation = ""
      try await Task.sleep(for: .milliseconds(30))
      #expect(editor.dictationOverlay == nil && overlay.superview == nil)
      #expect(editor.minSize.height == 0)
      #expect(heights.last == 72)
      #expect(editor.string == "A short draft.")
    }
  }

  @Test func presentationClearsStaleGhostsOnIdleFailureAndOtherModes() throws {
    var base: [String: Any] = [
      "mode": "dictate", "phase": "listening", "retryAvailable": false, "jobs": ["dictate"],
      "audioMode": false, "audioOk": true, "liveBlocked": false, "liveReason": "",
      "transcribers": [], "transcriber": "whisper", "devices": [], "device": "", "language": "sv",
      "languages": [], "session": "session", "dictation": [], "provisional": "Not yet final",
      "levels": [], "elapsed": 0, "remaining": 60, "limit": 60, "arming": false, "idle": false,
      "shouldStop": false, "error": "", "deviceNote": "",
    ]
    func value() throws -> String {
      try JSONDecoder().decode(
        StudioState.Audio.self, from: JSONSerialization.data(withJSONObject: base)
      ).composerProvisional
    }
    #expect(try value() == "Not yet final")
    base["phase"] = "finishing"
    #expect(try value() == "Not yet final")
    base["phase"] = "idle"
    #expect(try value().isEmpty)
    base["phase"] = "listening"
    base["error"] = "Socket lost"
    #expect(try value().isEmpty)
    base["error"] = ""
    base["mode"] = "live"
    #expect(try value().isEmpty)
    base["mode"] = "record"
    #expect(try value().isEmpty)
  }

  private func editor(_ initial: String) throws -> (
    NSHostingController<StudioDraftEditor>, DraftTextView
  ) {
    _ = NSApplication.shared
    var draft = initial
    let host = NSHostingController(
      rootView: StudioDraftEditor(
        text: Binding(get: { draft }, set: { draft = $0 }), onSend: {}))
    host.view.frame = NSRect(x: 0, y: 0, width: 360, height: 72)
    host.view.layoutSubtreeIfNeeded()
    return (host, try #require(find(host.view)))
  }
  private func find(_ view: NSView) -> DraftTextView? {
    (view as? DraftTextView) ?? view.subviews.lazy.compactMap(find).first
  }
}
