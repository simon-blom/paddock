import UserNotifications

struct DesktopNotificationSettings {
  var authorization: UNAuthorizationStatus
  var soundEnabled: Bool
  var showPreviews: UNShowPreviewsSetting = .whenAuthenticated
}

/// Keep OS permission dialogs and banners out of deterministic policy tests.
@MainActor protocol DesktopNotificationDelivering {
  func settings() async -> DesktopNotificationSettings
  func requestAuthorization() async throws
  func add(_ request: UNNotificationRequest) async throws
  func removeDelivered(_ identifiers: [String]?)
  func removePending()
}

@MainActor final class SystemNotificationDelivery: DesktopNotificationDelivering {
  let center = UNUserNotificationCenter.current()
  func settings() async -> DesktopNotificationSettings {
    let settings = await center.notificationSettings()
    return .init(
      authorization: settings.authorizationStatus, soundEnabled: settings.soundSetting == .enabled,
      showPreviews: settings.showPreviewsSetting)
  }
  func requestAuthorization() async throws {
    _ = try await center.requestAuthorization(options: [.alert, .sound])
  }
  func add(_ request: UNNotificationRequest) async throws { try await center.add(request) }
  func removeDelivered(_ identifiers: [String]?) {
    if let identifiers {
      center.removeDeliveredNotifications(withIdentifiers: identifiers)
    } else {
      center.removeAllDeliveredNotifications()
    }
  }
  func removePending() { center.removeAllPendingNotificationRequests() }
}
