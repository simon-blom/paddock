import AppKit
import Observation
import PaddockClient
import PaddockStudio
import UserNotifications

@MainActor @Observable
public final class DesktopNotifications: NSObject, UNUserNotificationCenterDelegate {
  public private(set) var authorization = "Not requested"
  public private(set) var error: String?
  public private(set) var requesting = false
  @ObservationIgnored private let center: UNUserNotificationCenter
  @ObservationIgnored private let defaults: UserDefaults
  @ObservationIgnored private var tracker = DesktopEventTracker()
  @ObservationIgnored public var open: ((DesktopAction) -> Void)?
  @ObservationIgnored public var isVisible: ((DesktopAction) -> Bool)?
  @ObservationIgnored private var pending: [String: (UUID, Task<Void, Never>)] = [:]
  public var enabled: Bool {
    didSet {
      defaults.set(enabled, forKey: "desktopNotifications")
      if !enabled {
        cancelPending()
        center.removeAllDeliveredNotifications()
      }
    }
  }
  public var sounds: Bool { didSet { defaults.set(sounds, forKey: "desktopNotificationSounds") } }

  public init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    enabled = defaults.bool(forKey: "desktopNotifications")
    sounds = defaults.bool(forKey: "desktopNotificationSounds")
    center = .current()
    super.init()
    center.delegate = self
    let open = UNNotificationAction(
      identifier: "open", title: "Open Paddock", options: [.foreground])
    center.setNotificationCategories([
      UNNotificationCategory(identifier: "paddock.activity", actions: [open], intentIdentifiers: [])
    ])
  }

  public func refreshAuthorization() async {
    let status = await center.notificationSettings().authorizationStatus
    switch status {
    case .authorized, .provisional: authorization = "Allowed by macOS"
    case .denied: authorization = "Disabled in macOS Notification Settings"
    case .notDetermined: authorization = "Not requested"
    default: authorization = "Managed by macOS"
    }
  }

  public func enable() async {
    guard !requesting else { return }
    requesting = true
    defer { requesting = false }
    do {
      error = nil
      enabled = try await center.requestAuthorization(options: [.alert, .sound])
      await refreshAuthorization()
    } catch { self.error = error.localizedDescription }
  }

  public func management(_ value: PaddockClient.ManagerSnapshot) {
    deliver(tracker.management(value))
  }
  public func studio(_ value: PaddockStudio.StudioState) { deliver(tracker.studio(value)) }

  private func deliver(_ events: [DesktopEvent]) {
    for event in events where enabled && isVisible?(event.route) != true {
      // Group approvals and compare lanes by conversation. Stable identifiers
      // replace the existing banner; they do not grow an unbounded history.
      let key = Self.identifier(event)
      pending[key]?.1.cancel()
      let generation = UUID()
      let task = Task { [weak self] in
        guard let self else { return }
        defer { if pending[key]?.0 == generation { pending[key] = nil } }
        let settings = await center.notificationSettings()
        guard !Task.isCancelled, enabled, isVisible?(event.route) != true,
          settings.authorizationStatus == .authorized
            || settings.authorizationStatus == .provisional
        else { return }
        let content = UNMutableNotificationContent()
        content.title = event.title
        content.body = event.body
        content.categoryIdentifier = "paddock.activity"
        content.threadIdentifier = key
        content.userInfo = Self.payload(event.route)
        if sounds && settings.soundSetting == .enabled { content.sound = .default }
        do {
          try await center.add(
            UNNotificationRequest(identifier: key, content: content, trigger: nil))
          if !enabled { center.removeDeliveredNotifications(withIdentifiers: [key]) }
        } catch { self.error = "Could not deliver a notification: \(error.localizedDescription)" }
      }
      pending[key] = (generation, task)
    }
  }

  private func cancelPending() {
    for task in pending.values { task.1.cancel() }
    pending.removeAll()
    center.removeAllPendingNotificationRequests()
  }

  static func identifier(_ event: DesktopEvent) -> String {
    switch event.route {
    case .conversation(let id): "paddock.\(event.kind == .approval ? "approval" : "reply").\(id)"
    case .chat(let port): "paddock.model.\(port)"
    default: "paddock.models"
    }
  }
  static func payload(_ route: DesktopAction) -> [String: String] {
    switch route {
    case .conversation(let id): ["destination": "conversation", "id": id]
    case .chat(let port): ["destination": "model", "port": String(port)]
    default: ["destination": "manager"]
    }
  }
  nonisolated static func route(_ payload: [AnyHashable: Any]) -> DesktopAction? {
    switch payload["destination"] as? String {
    case "manager": return .manager
    case "model":
      guard let value = payload["port"] as? String, let port = UInt16(value), port >= 1024 else {
        return nil
      }
      return .chat(port: port)
    case "conversation":
      guard let id = payload["id"] as? String, !id.isEmpty, id.utf8.count <= 128,
        id.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_") })
      else { return nil }
      return .conversation(id)
    default: return nil
    }
  }

  public nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse
  ) async {
    guard [UNNotificationDefaultActionIdentifier, "open"].contains(response.actionIdentifier),
      let route = Self.route(response.notification.request.content.userInfo)
    else { return }
    await MainActor.run { open?(route) }
  }
  public nonisolated func userNotificationCenter(
    _ center: UNUserNotificationCenter, willPresent notification: UNNotification
  ) async -> UNNotificationPresentationOptions {
    guard let route = Self.route(notification.request.content.userInfo) else { return [] }
    return await MainActor.run {
      guard enabled, isVisible?(route) != true else { return [] }
      return sounds ? [.banner, .list, .sound] : [.banner, .list]
    }
  }
}
