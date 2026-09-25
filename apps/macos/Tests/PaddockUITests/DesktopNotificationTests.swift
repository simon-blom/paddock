import Foundation
import PaddockClient
import PaddockNativeMarkdown
import Testing
import UserNotifications

@testable import PaddockConversationCore
@testable import PaddockStudio
@testable import PaddockUI

@Suite("Desktop notification delivery", .serialized) @MainActor
struct DesktopNotificationTests {
  @Test func previewIsBoundedPlainTextAndNeverContainsLinkTargetsOrMarkup() {
    #expect(
      NotificationExcerpt.text(
        "# Hello **world**\n\nUse `git status` and [the guide](https://secret.example/?token=private)."
      )
        == "Hello world Use git status and the guide.")
    #expect(NotificationExcerpt.text("```json\n{\"private\":1}\n```\n\nDone.") == "Done.")
    #expect(NotificationExcerpt.text("![secret image](https://secret.example/private)") == nil)
    #expect(NotificationExcerpt.text("<script>private</script>\n\nDone.") == "Done.")
    let unicode = NotificationExcerpt.text(String(repeating: "👨‍👩‍👧‍👦 café ", count: 1000))!
    #expect(unicode.count <= 160 && unicode.utf8.count <= 1003 && unicode.hasSuffix("…"))
    #expect(!unicode.contains("�"))
  }

  @Test func finalResponseSuppliesThePreviewNotThinkingOrPreviousTurns() async throws {
    let f = NotificationFixture()
    let running = try await state(busy: true, text: "streaming")
    f.notifications.studio(running)
    f.notifications.studio(
      try await state(busy: false, turn: "preview", text: "**Here is your answer.**"))
    await f.notifications.settle()
    let request = try #require(f.delivery.delivered.first)
    #expect(request.content.body == "Here is your answer.")
    #expect(request.content.title == "Your reply is ready")
    #expect(!String(describing: request.content.userInfo).contains("answer"))
    #expect(DesktopNotifications.route(request.content.userInfo) == .conversation("chat"))
  }

  @Test func previewsRespectBothInAppAndSystemPrivacySettings() async {
    for setting: UNShowPreviewsSetting in [.always, .whenAuthenticated, .never] {
      for enabled in [false, true] {
        let f = NotificationFixture()
        f.notifications.responsePreviews = enabled
        f.delivery.previews = setting
        var preview = event
        preview.preview = "Private response excerpt"
        f.notifications.deliver([preview])
        await f.notifications.settle()
        #expect(
          f.delivery.delivered.first?.content.body
            == (enabled && setting != .never ? preview.preview : preview.body))
        f.notifications.responsePreviews = false
        #expect(f.delivery.delivered.isEmpty)
        #expect(f.defaults.bool(forKey: "desktopNotificationPreviews") == false)
      }
    }
  }
  @Test func compareProducesOnePreviewAndNamesItsLane() async throws {
    var tracker = DesktopEventTracker()
    _ = tracker.studio(try await state(busy: true, text: "First answer", compare: true))
    let completed = try await state(
      busy: false, turn: "compare", text: "First answer", compare: true)
    let events = tracker.studio(completed)
    #expect(events.count == 1 && events[0].preview == "fixture: First answer")
    #expect(tracker.studio(completed).isEmpty)
  }

  @Test func emptyAnswersDoNotFallBackToPrivateThinking() async throws {
    var tracker = DesktopEventTracker()
    _ = tracker.studio(try await state(busy: true, text: ""))
    let events = tracker.studio(try await state(busy: false, turn: "empty", text: ""))
    #expect(events.count == 1 && events[0].preview == nil)
  }

  @Test func unavailableOrMismatchedTranscriptUsesTheGenericNotification() async throws {
    for otherConversation in [false, true] {
      var tracker = DesktopEventTracker()
      _ = tracker.studio(try await state(busy: true))
      let complete = try await state(
        busy: false, turn: "done", text: otherConversation ? "Wrong conversation" : nil,
        transcriptConversation: otherConversation ? "other" : "chat")
      let events = tracker.studio(complete)
      #expect(events.count == 1 && events.first?.preview == nil)
    }
  }
  @Test func firstReplyRequestsPermissionOnceAndNotAtLaunch() async throws {
    let f = NotificationFixture()
    #expect(f.notifications.enabled && f.delivery.requests == 0)
    f.notifications.studio(try await state(busy: false))
    await f.notifications.settle()
    #expect(f.delivery.requests == 0)
    f.notifications.studio(try await state(busy: true))
    await f.notifications.settle()
    #expect(f.delivery.requests == 1 && f.notifications.effectiveEnabled)
    f.notifications.studio(try await state(busy: false, turn: "reply-1"))
    await f.notifications.settle()
    #expect(f.delivery.delivered.count == 1)
    f.notifications.studio(try await state(busy: true, turn: "reply-1"))
    await f.notifications.settle()
    #expect(f.delivery.requests == 1)
  }

  @Test func readsIsAwayFromChatEvenWhileTheAppIsActive() async throws {
    let f = NotificationFixture()
    let workspace = WorkspaceModel()
    workspace.chat.apply(try await state(busy: true))
    let route = DesktopAction.conversation("chat")
    f.notifications.isVisible = {
      workspace.isDesktopDestinationVisible($0, workspaceActive: true)
    }
    for page: StudioDestination in [.newChat, .chats] {
      workspace.navigation.studio = page
      #expect(workspace.isDesktopDestinationVisible(route, workspaceActive: true))
      #expect(f.notifications.presentation(for: route).isEmpty)
    }
    workspace.navigation.studio = .reads
    #expect(!workspace.isDesktopDestinationVisible(route, workspaceActive: true))
    var tracker = DesktopEventTracker()
    _ = tracker.studio(try await state(busy: true))
    let terminal = try await state(busy: false, turn: "reply-1")
    f.notifications.deliver(tracker.studio(terminal))
    await f.notifications.settle()
    #expect(f.delivery.delivered.count == 1)
    #expect(f.notifications.presentation(for: route) == [.banner, .list])
    #expect(
      DesktopNotifications.route(f.delivery.delivered.first!.content.userInfo) == route)
    f.notifications.deliver(tracker.studio(terminal))
    await f.notifications.settle()
    #expect(f.delivery.delivered.count == 1)
    workspace.navigation.showSettings()
    #expect(!workspace.isDesktopDestinationVisible(route, workspaceActive: true))
    workspace.navigation.returnToChat()
    #expect(workspace.isDesktopDestinationVisible(route, workspaceActive: true))
    #expect(!workspace.isDesktopDestinationVisible(route, workspaceActive: false))
    #expect(!workspace.isDesktopDestinationVisible(.conversation("other"), workspaceActive: true))
  }

  @Test func explicitOptOutAndSystemDenialAreRespected() async {
    let off = NotificationFixture(enabled: false)
    off.notifications.prepareForBackgroundWork()
    off.notifications.deliver([event])
    await off.notifications.settle()
    #expect(off.delivery.requests == 0 && off.delivery.delivered.isEmpty)
    for deniedInitially in [true, false] {
      let f = NotificationFixture()
      f.delivery.status = deniedInitially ? .denied : .notDetermined
      f.delivery.grant = false
      for _ in 0..<3 {
        f.notifications.deliver([event])
        await f.notifications.settle()
      }
      #expect(f.delivery.requests == (deniedInitially ? 0 : 1))
      #expect(f.notifications.permissionBlocked && f.delivery.delivered.isEmpty)
    }
  }

  @Test func completionWaitsForPermissionAndRechecksVisibilityAndOptOut() async throws {
    for change in ["none", "visible", "disabled"] {
      let f = NotificationFixture()
      f.delivery.holdPermission = true
      var visible = false
      f.notifications.isVisible = { _ in visible }
      f.notifications.prepareForBackgroundWork()
      for _ in 0..<100 {
        if f.delivery.continuation != nil { break }
        await Task.yield()
      }
      #expect(f.delivery.continuation != nil)
      f.notifications.deliver([event, event])
      if change == "visible" { visible = true }
      if change == "disabled" { f.notifications.enabled = false }
      f.delivery.continuation?.resume()
      await f.notifications.settle()
      #expect(f.delivery.requests == 1)
      #expect(f.delivery.delivered.count == (change == "none" ? 1 : 0))
    }
  }

  @Test func permissionErrorsDoNotLoopAndExplicitEnableCanRetry() async {
    let f = NotificationFixture()
    f.delivery.fail = true
    f.notifications.prepareForBackgroundWork()
    await f.notifications.settle()
    #expect(f.notifications.error != nil && !f.notifications.requesting)
    f.notifications.deliver([event])
    await f.notifications.settle()
    #expect(f.delivery.requests == 1 && f.delivery.delivered.isEmpty)
    f.delivery.fail = false
    await f.notifications.enable()
    #expect(f.notifications.effectiveEnabled && f.notifications.error == nil)
  }

  private var event: DesktopEvent {
    .init(id: "reply-1", kind: .replyReady, route: .conversation("chat"))
  }

  private func state(
    busy: Bool, turn: String? = nil, text: String? = nil,
    transcriptConversation: String = "chat", compare: Bool = false
  ) async throws -> StudioState {
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: Data(
        #"{"origin":"http://127.0.0.1:43219","cookieName":"paddock_desktop_session","session":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#
          .utf8))
    let runtime = NativeStudioRuntime(transport: try NativeConversationTransport(host: host)) { _ in
    }
    var fields = await runtime.presentation()
    fields["revision"] = .number(1)
    fields["busy"] = .bool(busy)
    fields["conversation"] = .object([
      "id": .string("chat"), "title": .string("Private text stays out of notifications"),
      "model": .string("model"), "messageCount": .number(2),
    ])
    fields["activity"] = .object([
      "replies": .array([
        .object(["id": .string("reply"), "state": .string(busy ? "streaming" : "completed")])
      ]),
      "approvals": .array([]),
      "completedTurn": turn.map {
        .object([
          "id": .string($0), "conversationId": .string("chat"), "state": .string("completed"),
        ])
      } ?? .null,
    ])
    if compare, var activity = fields["activity"]?.object {
      activity["replies"] = .array([
        .object(["id": .string("reply"), "state": .string(busy ? "streaming" : "completed")]),
        .object(["id": .string("reply2"), "state": .string(busy ? "streaming" : "completed")]),
      ])
      fields["activity"] = .object(activity)
    }
    if let text {
      func message(_ id: String, _ role: String, _ text: String) -> ConversationValue {
        .object([
          "id": .string(id), "role": .string(role), "text": .string(text),
          "reasoning": .string("Private thinking must never be previewed"),
          "model": .string("fixture"), "streaming": .bool(false), "stopped": .bool(false),
          "error": .string(""), "incomplete": .bool(false),
          "group": compare && id.hasPrefix("reply") ? .string("comparison") : .null,
        ])
      }
      fields["nativeTranscript"] = .object([
        "available": .bool(true), "notice": .string(""),
        "conversationId": .string(transcriptConversation),
        "messages": .array(
          [
            message("old", "assistant", "Previous answer"),
            message("user", "user", "Private prompt"), message("reply", "assistant", text),
          ] + (compare ? [message("reply2", "assistant", "Second lane")] : [])),
      ])
    }
    return try JSONDecoder().decode(StudioState.self, from: JSONEncoder().encode(fields))
  }
}

@MainActor private final class NotificationFixture {
  let name = "PaddockNotificationTests.\(UUID().uuidString)"
  let defaults: UserDefaults
  let delivery = NotificationDeliveryFixture()
  let notifications: DesktopNotifications
  init(enabled: Bool? = nil) {
    defaults = UserDefaults(suiteName: name)!
    if let enabled { defaults.set(enabled, forKey: "desktopNotifications") }
    notifications = DesktopNotifications(defaults: defaults, delivery: delivery)
    notifications.isVisible = { _ in false }
  }
  deinit { UserDefaults(suiteName: name)?.removePersistentDomain(forName: name) }
}

@MainActor private final class NotificationDeliveryFixture: DesktopNotificationDelivering {
  var status: UNAuthorizationStatus = .notDetermined
  var grant = true
  var fail = false
  var holdPermission = false
  var continuation: CheckedContinuation<Void, Never>?
  var requests = 0
  var delivered: [UNNotificationRequest] = []
  var previews: UNShowPreviewsSetting = .whenAuthenticated
  func settings() async -> DesktopNotificationSettings {
    .init(authorization: status, soundEnabled: true, showPreviews: previews)
  }
  func requestAuthorization() async throws {
    requests += 1
    if holdPermission { await withCheckedContinuation { continuation = $0 } }
    if fail { throw CocoaError(.featureUnsupported) }
    status = grant ? .authorized : .denied
  }
  func add(_ request: UNNotificationRequest) async throws { delivered.append(request) }
  func removeDelivered(_ identifiers: [String]?) {
    if let identifiers {
      delivered.removeAll { identifiers.contains($0.identifier) }
    } else {
      delivered.removeAll()
    }
  }
  func removePending() {}
}
