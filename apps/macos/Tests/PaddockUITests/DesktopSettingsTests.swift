import AppKit
import ServiceManagement
import SwiftUI
import Testing
import UserNotifications

@testable import PaddockUI

@Suite("Application settings", .serialized) @MainActor
struct DesktopSettingsTests {
  @Test func notificationSwitchAndDeliveryUseActualPermission() {
    for state: UNAuthorizationStatus? in [nil, .notDetermined, .denied] {
      #expect(!DesktopNotifications.allowsDelivery(state))
    }
    for state: UNAuthorizationStatus in [.authorized, .provisional] {
      #expect(DesktopNotifications.allowsDelivery(state))
    }
  }
  @Test func readingLoginStateNeverRegistersOrInventsAnInstallWarning() throws {
    let fixture = try SettingsSystemFixture()
    for state: SMAppService.Status in [.notRegistered, .notFound, .enabled, .requiresApproval] {
      fixture.state = state
      fixture.preferences.refreshLogin()
      #expect(fixture.preferences.loginEnabled == (state == .enabled || state == .requiresApproval))
      #expect(fixture.preferences.loginNeedsApproval == (state == .requiresApproval))
      #expect(fixture.preferences.loginError == nil)
    }
    #expect(fixture.operations.isEmpty)
  }

  @Test func missingLoginRegistrationCanBeEnabledWithoutAssumingAnInstallPath() async throws {
    let fixture = try SettingsSystemFixture()
    fixture.state = .notFound
    fixture.preferences.refreshLogin()
    await fixture.preferences.setLogin(true)
    #expect(fixture.operations == ["register"])
    #expect(fixture.preferences.loginEnabled && !fixture.preferences.loginBusy)
    #expect(fixture.preferences.loginError == nil)
  }

  @Test func awaitingApprovalCanBeCancelledAndExternalChangesAreReflected() async throws {
    let fixture = try SettingsSystemFixture()
    fixture.registeredState = .requiresApproval
    await fixture.preferences.setLogin(true)
    #expect(fixture.preferences.loginEnabled && fixture.preferences.loginNeedsApproval)
    await fixture.preferences.setLogin(false)
    #expect(fixture.operations == ["register", "unregister"])
    #expect(!fixture.preferences.loginEnabled && !fixture.preferences.loginNeedsApproval)
    fixture.state = .enabled
    fixture.preferences.refreshLogin()
    #expect(fixture.preferences.loginEnabled)
    fixture.state = .notRegistered
    fixture.preferences.refreshLogin()
    #expect(!fixture.preferences.loginEnabled)
  }

  @Test func failedLoginOperationsDoNotPretendToSucceed() async throws {
    let fixture = try SettingsSystemFixture()
    fixture.fail = true
    await fixture.preferences.setLogin(true)
    #expect(!fixture.preferences.loginEnabled && !fixture.preferences.loginBusy)
    #expect(fixture.preferences.loginError?.contains("Synthetic failure") == true)
    fixture.preferences.refreshLogin()
    #expect(fixture.preferences.loginError != nil)
    fixture.fail = false
    await fixture.preferences.setLogin(true)
    #expect(fixture.preferences.loginEnabled && fixture.preferences.loginError == nil)
    fixture.fail = true
    await fixture.preferences.setLogin(false)
    #expect(fixture.preferences.loginEnabled && fixture.preferences.loginError != nil)
    fixture.state = .notRegistered
    fixture.preferences.refreshLogin()
    #expect(!fixture.preferences.loginEnabled && fixture.preferences.loginError == nil)
  }

  @Test func inconsistentSystemAcknowledgementIsAnError() async throws {
    let fixture = try SettingsSystemFixture()
    fixture.registeredState = .notFound
    await fixture.preferences.setLogin(true)
    #expect(!fixture.preferences.loginEnabled)
    #expect(fixture.preferences.loginError != nil)
  }

  @Test func shortcutToggleRemembersChoiceAcrossDisableAndAppRestart() throws {
    let fixture = try SettingsSystemFixture()
    let preferences = fixture.preferences
    #expect(!preferences.shortcutEnabled && fixture.shortcuts.isEmpty)
    preferences.setShortcutEnabled(true)
    #expect(preferences.shortcut == "control-option-space")
    preferences.configureShortcut("command-option-j")
    preferences.setShortcutEnabled(false)
    #expect(!preferences.shortcutEnabled)
    let restored = fixture.makePreferences()
    #expect(!restored.shortcutEnabled)
    restored.setShortcutEnabled(true)
    #expect(restored.shortcut == "command-option-j")
    #expect(
      fixture.shortcuts == ["control-option-space", "command-option-j", "off", "command-option-j"])
    #expect(!QuestionHotKey.choices.contains { $0.1 == "Off" })
  }

  @Test func shortcutConflictPreservesSelectionAndSavedPreference() throws {
    let fixture = try SettingsSystemFixture()
    fixture.preferences.configureShortcut("command-option-j")
    fixture.fail = true
    fixture.preferences.configureShortcut("command-shift-space")
    #expect(fixture.preferences.shortcut == "command-option-j")
    #expect(fixture.preferences.shortcutError != nil)
    #expect(fixture.defaults.string(forKey: "desktopQuestionShortcut") == "command-option-j")
    fixture.fail = false
    fixture.preferences.setShortcutEnabled(false)
    #expect(fixture.preferences.shortcutError == nil)
  }

  @Test func applicationSettingsFitBothThemesAndNarrowWidthsWithoutOSPrompts() throws {
    for dark in [false, true] {
      for width: CGFloat in [380, 680] {
        for needsAction in [false, true] {
          let fixture = try SettingsSystemFixture()
          fixture.state = needsAction ? .requiresApproval : .notFound
          fixture.preferences.refreshLogin()
          if needsAction { fixture.preferences.setShortcutEnabled(true) }
          let content = DesktopSettingsSurface(embedded: true) {
            DesktopSettingsContent(
              preferences: fixture.preferences, notify: .constant(false), sounds: .constant(false),
              previews: .constant(true),
              notificationBusy: false, notificationBlocked: needsAction, notificationError: nil,
              openNotifications: { Issue.record("Layout must not open System Settings") })
          }
          let host = NSHostingView(
            rootView: content.frame(width: width, height: 850)
              .environment(\.colorScheme, dark ? .dark : .light))
          host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
          host.frame = NSRect(x: 0, y: 0, width: width, height: 850)
          host.layoutSubtreeIfNeeded()
          #expect(host.fittingSize.width == width)
          #expect(fixture.operations.isEmpty)
          if let folder = ProcessInfo.processInfo.environment["PADDOCK_SETTINGS_SNAPSHOTS"],
            let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)
          {
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let url = URL(fileURLWithPath: folder)
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
            try bitmap.representation(using: .png, properties: [:])?.write(
              to: url.appending(
                path: "application-\(Int(width))-\(dark ? "dark" : "light")-\(needsAction).png"))
          }
        }
      }
    }
  }
}

@MainActor private final class SettingsSystemFixture {
  let domain = "io.truespar.paddock.application-settings-test.\(UUID())"
  let defaults: UserDefaults
  var state = SMAppService.Status.notRegistered
  var registeredState = SMAppService.Status.enabled
  var fail = false
  var operations: [String] = []
  var shortcuts: [String] = []
  lazy var preferences = makePreferences()

  init() throws { defaults = try #require(UserDefaults(suiteName: domain)) }

  func makePreferences() -> DesktopPreferences {
    DesktopPreferences(
      defaults: defaults,
      loginService: DesktopLoginService(
        status: { [unowned self] in state },
        register: { [unowned self] in
          operations.append("register")
          if fail { throw SyntheticFailure() }
          state = registeredState
        },
        unregister: { [unowned self] in
          operations.append("unregister")
          if fail { throw SyntheticFailure() }
          state = .notRegistered
        }),
      registerShortcut: { [unowned self] value, _ in
        if fail { throw SyntheticFailure() }
        shortcuts.append(value)
      })
  }
  isolated deinit { defaults.removePersistentDomain(forName: domain) }
}

private struct SyntheticFailure: LocalizedError {
  var errorDescription: String? { "Synthetic failure" }
}
