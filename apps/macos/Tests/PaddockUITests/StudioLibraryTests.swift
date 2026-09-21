import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native prompts and preferences", .serialized) @MainActor
struct StudioLibraryTests {
  @Test func saveAcknowledgementAndFailureKeepTheRightDraft() async {
    let fixture = LibraryFixture()
    let model = StudioLibraryModel()
    model.command = fixture.command
    model.create()
    model.editor?.name = "Review"
    model.editor?.body = "Check the evidence."
    fixture.fail = true
    model.save()
    await model.settle()
    #expect(model.dirty && model.editor?.body == "Check the evidence." && model.error != nil)
    fixture.fail = false
    model.save()
    #expect(model.saving)
    await model.settle()
    #expect(!model.dirty && model.editor?.revision == "saved" && model.error == nil)
    #expect(fixture.calls.last?.1["revision"] == .string(""))
  }
  @Test func openingAnotherPresetCannotReplaceAnUnsavedDraft() async {
    let f = LibraryFixture()
    let m = StudioLibraryModel()
    m.command = f.command
    m.create()
    m.editor?.body = "Keep this"
    await m.open("other")
    #expect(f.calls.isEmpty && m.editor?.body == "Keep this")
    m.discard()
    await m.open("p")
    #expect(m.editor?.revision == "original" && !m.dirty)
  }
  @Test func failedDeleteLeavesTheReviewedRecord() async {
    let f = LibraryFixture()
    let m = StudioLibraryModel()
    m.command = f.command
    await m.open("p")
    f.fail = true
    m.remove()
    await m.settle()
    #expect(m.editor?.id == "p" && m.error != nil)
    #expect(f.calls.last?.1["revision"] == .string("original"))
  }
  @Test func invalidAndMultibytePresetsNeverWrite() {
    let m = StudioLibraryModel()
    m.create()
    m.save()
    #expect(!m.saving && m.validation != nil)
    m.editor?.name = "Valid"
    m.editor?.body = String(repeating: "🙂", count: 40_000)
    #expect(m.validation != nil)
  }
  @Test func studioNavigationKeepsAllEditorsAndQuitProtectsThem() {
    let workspace = WorkspaceModel()
    workspace.studioLibrary.create()
    workspace.studioLibrary.editor?.body = "Keep across Manager"
    workspace.navigation.studio = .prompts
    workspace.navigation.showManager(.runners)
    #expect(workspace.studioNeedsQuitConfirmation)
    workspace.navigation.mode = .studio
    #expect(workspace.navigation.studio == .prompts)
    #expect(workspace.studioLibrary.editor?.body == "Keep across Manager")
    workspace.studioLibrary.discard()
    #expect(!workspace.studioNeedsQuitConfirmation)
  }
  @Test func preferencesSendOnlyChangedFieldsAndKeepFailedDraft() async {
    let f = LibraryFixture()
    let m = StudioPreferencesModel()
    m.command = f.command
    await m.load()
    #expect(!m.dirty)
    m.toolLimit = "50"
    f.fail = true
    m.save()
    await m.settle()
    #expect(m.dirty && m.toolLimit == "50")
    let patch = f.calls.last?.1["changes"]?.object
    #expect(patch == ["maxToolCalls": .number(50)])
    #expect(f.calls.last?.1["expected"]?.object == ["maxToolCalls": .null])
    await m.load()
    #expect(m.toolLimit == "50")  // Navigation cannot discard a failed draft.
    f.fail = false
    m.save()
    await m.settle()
    #expect(!m.dirty && m.error == nil && m.toolLimit == "50")
  }
  @Test func invalidPreferenceTextIsNotInterpretedAsDefault() async {
    let f = LibraryFixture()
    let m = StudioPreferencesModel()
    m.command = f.command
    await m.load()
    m.replyLimit = "abc"
    m.save()
    await m.settle()
    #expect(m.validation != nil && !m.saving && f.calls.count == 1)
  }
  @Test func settingsUseWebOrderAndChoicesWithoutRewritingAnOversizedSavedLimit() async throws {
    let f = LibraryFixture()
    f.prefs["maxTokens"] = .number(32768)
    let m = StudioPreferencesModel()
    m.command = f.command
    await m.load()
    let layout = try #require(m.layout)
    #expect(
      layout.sections.map(\.id) == [
        "maxTokens", "maxToolCalls", "summarize", "microphone", "mapTiles",
      ])
    #expect(layout.toolStops.map(\.value) == [0, 5, 10, 25, 50, 100])
    #expect(layout.replyIndex(m.replyLimit) == layout.replyStops.count - 1)
    #expect(m.replyLimit == "32768" && !m.dirty)
    m.summarize = false
    m.save()
    await m.settle()
    #expect(f.calls.last?.1["changes"]?.object == ["summarize": .bool(false)])
    #expect(m.replyLimit == "32768" && !m.dirty)
    #expect(f.prefs["autoTitle"] == .bool(true) && f.prefs["markUnsure"] == .bool(true))
  }
  @Test func malformedStoredLimitIsReportedWithoutNativeIntegerOverflow() async {
    let f = LibraryFixture()
    let m = StudioPreferencesModel()
    f.prefs["maxTokens"] = .number(1e100)
    m.command = f.command
    await m.load()
    #expect(!m.loaded && m.error != nil)
  }
  @Test func instructionsKeepDraftAcrossCloseAndOnlyApplyToReviewedConversation() async {
    let f = LibraryFixture()
    let m = StudioInstructionsModel()
    await m.load(command: f.command)
    m.body = "New instructions"
    await m.load(command: f.command)
    #expect(m.body == "New instructions" && f.calls.count == 1)
    f.fail = true
    m.apply(command: f.command)
    await m.settle()
    #expect(m.dirty && m.body == "New instructions" && m.error != nil)
    #expect(f.calls.last?.1["conversationId"] == .string("chat-fixture"))
    #expect(f.calls.last?.1["expected"] == .string("Previous"))
    f.fail = false
    m.apply(command: f.command)
    await m.settle()
    #expect(!m.dirty && m.error == nil)
  }
  @Test func nativeSurfacesFitOffscreenInBothThemes() async throws {
    for dark in [false, true] {
      let f = LibraryFixture()
      let library = StudioLibraryModel()
      let settings = StudioPreferencesModel()
      library.command = f.command
      settings.command = f.command
      await library.open("p")
      await settings.load()
      for (name, content) in [
        ("prompts", AnyView(StudioLibraryView(model: library))),
        ("preferences", AnyView(StudioPreferencesView(model: settings, busy: false))),
      ] {
        let host = NSHostingView(
          rootView: content.frame(width: 680, height: 780).environment(
            \.colorScheme, dark ? .dark : .light))
        host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        host.frame = NSRect(x: 0, y: 0, width: 680, height: 780)
        host.layoutSubtreeIfNeeded()
        #expect(host.fittingSize.width == 680)
        if let folder = ProcessInfo.processInfo.environment["PADDOCK_PROMPT_SNAPSHOTS"],
          let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)
        {
          host.cacheDisplay(in: host.bounds, to: bitmap)
          let url = URL(fileURLWithPath: folder)
          try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
          try bitmap.representation(using: .png, properties: [:])?.write(
            to: url.appending(path: "\(name)-\(dark ? "dark" : "light").png"))
        }
      }
    }
  }
}

@MainActor private final class LibraryFixture {
  var fail = false
  var calls: [(String, [String: StudioValue])] = []
  var prefs: [String: StudioValue] = [
    "maxTokens": .null, "maxToolCalls": .null, "summarize": .bool(true), "autoTitle": .bool(true),
    "markUnsure": .bool(true), "mapTiles": .string(""), "mapHost": .string("tiles.openfreemap.org"),
  ]
  func command(_ kind: String, _ p: [String: StudioValue]) async throws -> [String: StudioValue] {
    calls.append((kind, p))
    if fail { throw StudioLibraryError.message("Synthetic save failure") }
    if kind == "promptGet" {
      return [
        "prompt": .object([
          "id": .string("p"), "name": .string("Evidence review"),
          "body": .string("Review the evidence, state uncertainties and cite sources."),
          "revision": .string("original"),
        ])
      ]
    }
    if kind == "promptSave" {
      var record = p
      record["revision"] = .string("saved")
      return ["prompt": .object(record)]
    }
    if kind == "promptList" {
      return [
        "library": .object([
          "rows": .array([]), "page": .number(0), "pageSize": .number(40), "total": .number(0),
          "matched": .number(0),
        ])
      ]
    }
    if kind == "preferencesSave", let changes = p["changes"]?.object {
      prefs.merge(changes) { _, new in new }
    }
    if kind == "preferencesSave" || kind == "preferencesGet" {
      var result = prefs
      var root = URL(fileURLWithPath: #filePath)
      for _ in 0..<5 { root.deleteLastPathComponent() }
      let data = try Data(
        contentsOf: root.appending(path: "studio/src/lib/studio-settings-layout.fixture.json"))
      result["layout"] = try JSONDecoder().decode(StudioValue.self, from: data)
      return ["preferences": .object(result)]
    }
    if kind == "instructionsGet" {
      return [
        "instructions": .object([
          "conversationId": .string("chat-fixture"), "body": .string("Previous"),
          "blocks": .array([]),
        ])
      ]
    }
    return [:]
  }
}
