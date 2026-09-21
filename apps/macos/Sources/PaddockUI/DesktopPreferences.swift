import AppKit
import Carbon
import Observation
import ServiceManagement
import SwiftUI

@MainActor @Observable
public final class DesktopPreferences {
  public var showMenuBar: Bool {
    didSet {
      defaults.set(showMenuBar, forKey: "desktopShowMenuBar")
      onMenuBarChange?(showMenuBar)
    }
  }
  public private(set) var shortcut: String
  public private(set) var shortcutError: String?
  public private(set) var loginStatus: String? = "Checking…"
  public private(set) var loginEnabled = false
  public private(set) var loginBusy = false
  public private(set) var loginError: String?
  @ObservationIgnored private let defaults: UserDefaults
  @ObservationIgnored private let hotKey = QuestionHotKey()
  @ObservationIgnored public var onQuestion: (() -> Void)?
  @ObservationIgnored public var onMenuBarChange: ((Bool) -> Void)?

  public init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    showMenuBar = defaults.object(forKey: "desktopShowMenuBar") as? Bool ?? true
    shortcut = defaults.string(forKey: "desktopQuestionShortcut") ?? "off"
  }
  public func start() {
    configureShortcut(shortcut)
    refreshLogin()
  }
  public func configureShortcut(_ selection: String) {
    do {
      try hotKey.register(selection) { [weak self] in self?.onQuestion?() }
      shortcut = selection
      defaults.set(selection, forKey: "desktopQuestionShortcut")
      shortcutError = nil
    } catch { shortcutError = error.localizedDescription }
  }
  public func refreshLogin() {
    switch SMAppService.mainApp.status {
    case .enabled:
      loginEnabled = true
      loginStatus = nil
    case .requiresApproval:
      loginEnabled = false
      loginStatus = "Allow Paddock in System Settings > General > Login Items."
    case .notRegistered:
      loginEnabled = false
      loginStatus = nil
    case .notFound:
      loginEnabled = false
      loginStatus = "Install Paddock in Applications before enabling launch at login."
    @unknown default:
      loginEnabled = false
      loginStatus = "Status unavailable"
    }
  }
  public func setLogin(_ enabled: Bool) async {
    guard !loginBusy else { return }
    loginBusy = true
    defer {
      loginBusy = false
      refreshLogin()
    }
    do {
      loginError = nil
      if enabled {
        try SMAppService.mainApp.register()
      } else {
        try await SMAppService.mainApp.unregister()
      }
    } catch { loginError = error.localizedDescription }
  }
}

/// Register only the selected chord. Unlike a global event monitor, this does
/// not observe typing and does not require Accessibility/Input Monitoring.
/// Registration failure leaves the previous shortcut intact.
@MainActor final class QuestionHotKey {
  private var reference: EventHotKeyRef?
  private var handler: EventHandlerRef?
  private var callback: (() -> Void)?
  private var selection = "off"
  private var serial: UInt32 = 0
  static let choices = [
    ("off", "Off"), ("control-option-space", "⌃⌥Space"), ("command-shift-space", "⇧⌘Space"),
    ("command-option-j", "⌥⌘J"),
  ]

  func register(_ value: String, action: @escaping () -> Void) throws {
    guard Self.choices.contains(where: { $0.0 == value }) else {
      throw DesktopShortcutError.invalid
    }
    if value == selection {
      callback = action
      return
    }
    if value == "off" {
      if let reference { UnregisterEventHotKey(reference) }
      reference = nil
      callback = nil
      selection = value
      return
    }
    if handler == nil {
      var type = EventTypeSpec(
        eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
      let status = InstallEventHandler(
        GetApplicationEventTarget(),
        { _, event, context in
          guard let event, let context else { return OSStatus(eventNotHandledErr) }
          var id = EventHotKeyID()
          guard
            GetEventParameter(
              event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
              nil,
              MemoryLayout<EventHotKeyID>.size, nil, &id) == noErr
          else { return OSStatus(eventNotHandledErr) }
          // Application-target Carbon events are dispatched on the main run loop.
          return MainActor.assumeIsolated {
            let owner = Unmanaged<QuestionHotKey>.fromOpaque(context).takeUnretainedValue()
            guard id.signature == 0x5041_4451, id.id == owner.serial else {
              return OSStatus(eventNotHandledErr)
            }
            owner.callback?()
            return noErr
          }
        }, 1, &type, Unmanaged.passUnretained(self).toOpaque(), &handler)
      guard status == noErr else { throw DesktopShortcutError.unavailable }
    }
    let code = value == "command-option-j" ? UInt32(kVK_ANSI_J) : UInt32(kVK_Space)
    let modifiers =
      value == "control-option-space"
      ? controlKey | optionKey
      : value == "command-shift-space" ? cmdKey | shiftKey : cmdKey | optionKey
    var candidate: EventHotKeyRef?
    let next = serial &+ 1
    let status = RegisterEventHotKey(
      code, UInt32(modifiers), EventHotKeyID(signature: 0x5041_4451, id: next),
      GetApplicationEventTarget(), OptionBits(kEventHotKeyExclusive), &candidate)
    guard status == noErr else { throw DesktopShortcutError.unavailable }
    if let reference { UnregisterEventHotKey(reference) }
    reference = candidate
    serial = next
    selection = value
    callback = action
  }

  isolated deinit {
    if let reference { UnregisterEventHotKey(reference) }
    if let handler { RemoveEventHandler(handler) }
  }
}

enum DesktopShortcutError: LocalizedError {
  case invalid, unavailable
  var errorDescription: String? {
    self == .invalid
      ? "Choose a supported shortcut."
      : "This shortcut could not be registered. It may be used by another app. Choose another; the previous shortcut was kept."
  }
}

public struct DesktopSettingsView: View {
  @Bindable var preferences: DesktopPreferences
  @Bindable var notifications: DesktopNotifications
  var embedded: Bool
  public init(
    preferences: DesktopPreferences, notifications: DesktopNotifications, embedded: Bool = false
  ) {
    self.preferences = preferences
    self.notifications = notifications
    self.embedded = embedded
  }
  public var body: some View {
    DesktopSettingsSurface(embedded: embedded) {
      Section("System integration") {
        Toggle("Show Paddock in the menu bar", isOn: $preferences.showMenuBar)
        Toggle(
          "Launch Paddock at login",
          isOn: Binding(
            get: { preferences.loginEnabled },
            set: { value in Task { await preferences.setLogin(value) } })
        )
        .disabled(preferences.loginBusy)
        if let status = preferences.loginStatus {
          Text(status).font(.caption).foregroundStyle(.secondary)
        }
        if let error = preferences.loginError { Text(error).foregroundStyle(PaddockStyle.caution) }
        Button("Open Login Items Settings") { SMAppService.openSystemSettingsLoginItems() }
        Dropdown(
          title: "Quick Question shortcut",
          value: QuestionHotKey.choices.first(where: { $0.0 == preferences.shortcut })?.1 ?? "Off"
        ) {
          ForEach(QuestionHotKey.choices, id: \.0) { choice in
            Button(choice.1) { preferences.configureShortcut(choice.0) }
          }
        }
        if let error = preferences.shortcutError {
          Text(error).foregroundStyle(PaddockStyle.caution)
        }
      }.listRowBackground(PaddockStyle.surface)
      Section("Notifications") {
        Toggle(
          "Notify me about background work",
          isOn: Binding(
            get: { notifications.enabled },
            set: { value in
              if value {
                Task { await notifications.enable() }
              } else {
                notifications.enabled = false
              }
            })
        ).disabled(notifications.requesting)
        Toggle("Play notification sounds", isOn: $notifications.sounds).disabled(
          !notifications.enabled)
        Text(notifications.authorization).font(.caption).foregroundStyle(.secondary)
        if let error = notifications.error { Text(error).foregroundStyle(PaddockStyle.caution) }
      }.listRowBackground(PaddockStyle.surface)
    }
    .task {
      preferences.refreshLogin()
      await notifications.refreshAuthorization()
    }
    .onReceive(
      NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)
    ) { _ in
      preferences.refreshLogin()
      Task { await notifications.refreshAuthorization() }
    }
  }
}

/// The Settings scene and its grouped form are separate paint surfaces. Keep
/// both opaque; changing the scroll background alone leaves a tinted title bar.
struct DesktopSettingsSurface<Content: View>: View {
  @AppStorage("workspaceAppearance") private var appearance: WorkspaceAppearance = .system
  var embedded = false
  @ViewBuilder var content: Content
  var body: some View {
    if embedded {
      VStack(alignment: .leading, spacing: 0) {
        PageHeading(title: "Application") { EmptyView() }.padding(.horizontal, 32).padding(.top, 32)
        PaddockScrollRegion {
          Form { content }.formStyle(.grouped).scrollContentBackground(.hidden)
        }
      }.frame(maxWidth: 820, maxHeight: .infinity).frame(maxWidth: .infinity)
        .background(PaddockStyle.canvas).buttonStyle(FlatButtonStyle())
    } else {
      PaddockScrollRegion {
        Form { content }.formStyle(.grouped).scrollContentBackground(.hidden)
      }.frame(width: 530, height: 580)
        .desktopWindowSurface(appearance: appearance).buttonStyle(FlatButtonStyle())
        // Settings' SwiftUI scene reapplies its native title-bar treatment after
        // hosting. Declare the toolbar paint too, instead of racing its window setup.
        .toolbarBackground(PaddockStyle.canvas, for: .windowToolbar)
        .toolbarBackgroundVisibility(.visible, for: .windowToolbar)
        .preferredColorScheme(appearance.colorScheme)
    }
  }
}
