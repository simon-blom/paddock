import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockConversationCore
@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native conversation behavior", .serialized) @MainActor
struct NativeConversationBehaviorTests {
  @Test func thinkingFoldsWhenAnswerStartsLikeWebStudio() async throws {
    _ = NSApplication.shared
    let reasoning = (0..<60).map { "Reasoning step \($0), visible as it arrives." }.joined(
      separator: "\n\n")
    let host = NSHostingController(
      rootView: NativeThinkingBlock(message: try message(reasoning: reasoning, streaming: true)))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 600, height: 320), styleMask: [.borderless],
      backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    defer { window.close() }
    for _ in 0..<100 {
      host.view.layoutSubtreeIfNeeded()
      if texts(host.view).contains(where: { $0.string.contains("Reasoning step 59") }),
        (find(NSScrollView.self, host.view)?.bounds.height ?? 0) > 100
      {
        break
      }
      try await Task.sleep(for: .milliseconds(20))
    }
    #expect(texts(host.view).contains { $0.string.contains("Reasoning step 59") })
    let scroll = try #require(find(NSScrollView.self, host.view))
    #expect(
      scroll.bounds.height <= 221 && scroll.bounds.height > 100,
      "Live reasoning viewport: \(scroll.bounds.height)")
    host.rootView = NativeThinkingBlock(
      message: try message(reasoning: reasoning, text: "Answer has begun", streaming: true))
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    #expect(texts(host.view).isEmpty)
    host.rootView = NativeThinkingBlock(
      message: try message(reasoning: reasoning, text: "Answer finished", streaming: false))
    try await Task.sleep(for: .milliseconds(100))
    host.view.layoutSubtreeIfNeeded()
    #expect(texts(host.view).isEmpty)
  }

  @Test func htmlAutoOpensCloseSurvivesUpdatesAndRevisitRestoresAvailability() async throws {
    let raw: [String: String] = [
      "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
      "session": String(repeating: "a", count: 64),
    ]
    let host = try JSONDecoder().decode(StudioHost.self, from: JSONEncoder().encode(raw))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in
    }
    let workspace = StudioWorkspace(client: NoCore())
    for (revision, cid, open, count, expected): (Int, String, Bool, Int, String?) in [
      (1, "first", true, 0, nil),
      (2, "first", true, 1, "art_012345abcdef"),
      (3, "first", false, 1, nil),
      (4, "first", false, 2, nil),
      (5, "other", true, 0, nil),
      (6, "first", true, 2, "art_012345abcdef"),
    ] {
      let fields = try await runtime.artifactBehaviorFixture(
        id: cid, open: open, count: count, revision: revision)
      workspace.apply(
        try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields)))
      #expect(workspace.selectedArtifactId == expected)
      #expect(workspace.state?.nativeArtifacts?.count == count)
    }
    await runtime.close()
    await workspace.shutdown()
  }
  @Test func closingComparePreviewLeavesOtherWritersOpenAndShowRestoresOnlyItsArtifact()
    async throws
  {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in
    }
    let workspace = StudioWorkspace(client: NoCore())
    func snapshot(_ revision: Int, count: Int = 2, conversation: String = "compare") async throws {
      let fields = try await runtime.artifactBehaviorFixture(
        id: conversation, open: true, count: count, revision: revision)
      workspace.apply(
        try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields)))
    }
    try await snapshot(1)
    let first = "art_012345abcdef"
    let second = "art_abcdef012345"
    workspace.revealArtifact(first)
    workspace.artifactDrafts[first] = .init(saved: "original")
    workspace.artifactDrafts[first]?.text = "unsaved edit"
    workspace.dismissArtifact(first)
    #expect(workspace.presentedArtifacts.map(\.id) == [second])
    #expect(workspace.selectedArtifactId == second)
    #expect(workspace.artifactDrafts[first]?.text == "unsaved edit")
    try await snapshot(2)
    #expect(
      workspace.presentedArtifacts.map(\.id) == [second],
      "Streaming refresh must not reopen a closed preview")
    workspace.dismissArtifact(second)
    #expect(workspace.presentedArtifacts.isEmpty && workspace.selectedArtifactId == nil)
    try await snapshot(3)
    #expect(workspace.presentedArtifacts.isEmpty && workspace.selectedArtifactId == nil)
    workspace.revealArtifact(first)
    #expect(
      workspace.presentedArtifacts.map(\.id) == [first], "Show reopens only the chosen artifact")
    #expect(workspace.selectedArtifactId == first)
    #expect(workspace.artifactPicks["writer-0"] == first)
    workspace.revealArtifact("not-in-this-conversation")
    workspace.dismissArtifact("not-in-this-conversation")
    #expect(workspace.selectedArtifactId == first)
    try await snapshot(4, count: 0, conversation: "other")
    try await snapshot(5)
    #expect(
      workspace.presentedArtifacts.count == 2, "Revisiting retains access to every stored artifact")
    #expect(workspace.artifactDrafts[first]?.text == "unsaved edit")
    await runtime.close()
    await workspace.shutdown()
  }

  @Test func newArtifactCanOpenWhileAnEarlierPreviewStaysDismissed() async throws {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:43219", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in
    }
    let workspace = StudioWorkspace(client: NoCore())
    for count in [1, 2] {
      let fields = try await runtime.artifactBehaviorFixture(
        id: "same", open: true, count: count, revision: count)
      workspace.apply(
        try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields)))
      if count == 1 { workspace.dismissArtifact("art_012345abcdef") }
    }
    #expect(workspace.presentedArtifacts.map(\.id) == ["art_abcdef012345"])
    #expect(workspace.selectedArtifactId == "art_abcdef012345")
    await runtime.close()
    await workspace.shutdown()
  }
  private struct NoCore: ManagerLoading {
    func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
  }
  private func message(reasoning: String, text: String = "", streaming: Bool) throws
    -> StudioState.NativeTranscript.Message
  {
    try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": "thinking", "role": "assistant", "model": "fixture", "text": text,
        "reasoning": reasoning,
        "streaming": streaming, "stopped": false, "error": "", "incomplete": false,
      ]))
  }
  private func texts(_ root: NSView) -> [NSTextView] {
    ((root as? NSTextView).map { [$0] } ?? []) + root.subviews.flatMap(texts)
  }
  private func find<T: NSView>(_ type: T.Type, _ root: NSView) -> T? {
    (root as? T) ?? root.subviews.lazy.compactMap { find(type, $0) }.first
  }
}

extension NativeStudioRuntime {
  fileprivate func artifactBehaviorFixture(id: String, open: Bool, count: Int, revision: Int) throws
    -> O
  {
    document = try .init(fields: [
      "id": .string(id), "title": .string("Artifact fixture"), "model": .string("local"),
      "messages": .array([]), "artifactsPaneOpen": .bool(open),
    ])
    draft = false
    self.revision = revision
    artifacts = (0..<count).map { i in
      .object([
        "id": .string(i == 0 ? "art_012345abcdef" : "art_abcdef012345"), "kind": .string("html"),
        "title": .string("Page"), "model": .string("writer-\(i)"), "versions": .number(1),
        "updatedAt": .number(1),
      ])
    }
    return presentation()
  }
}
