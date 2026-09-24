import AppKit
import Carbon
import Observation
import ServiceManagement

@MainActor struct DesktopLoginService {
  var status: () -> SMAppService.Status
  var register: () throws -> Void
  var unregister: () async throws -> Void

  static var system: Self {
    Self(
      status: { SMAppService.mainApp.status }, register: { try SMAppService.mainApp.register() },
      unregister: { try await SMAppService.mainApp.unregister() })
  }
}

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
  private(set) var loginState: SMAppService.Status?
  // A registered item awaiting permission stays selected so it can also be removed.
  public var loginEnabled: Bool { loginState == .enabled || loginState == .requiresApproval }
  public var loginNeedsApproval: Bool { loginState == .requiresApproval }
  public private(set) var loginBusy = false
  public private(set) var loginError: String?
  @ObservationIgnored private let defaults: UserDefaults
  @ObservationIgnored private let loginService: DesktopLoginService
  @ObservationIgnored private let registerShortcut: (String, @escaping () -> Void) throws -> Void
  private var rememberedShortcut: String
  @ObservationIgnored public var onQuestion: (() -> Void)?
  @ObservationIgnored public var onMenuBarChange: ((Bool) -> Void)?

  public convenience init(defaults: UserDefaults = .standard) {
    self.init(
      defaults: defaults, loginService: .system, registerShortcut: QuestionHotKey().register)
  }

  init(
    defaults: UserDefaults, loginService: DesktopLoginService,
    registerShortcut: @escaping (String, @escaping () -> Void) throws -> Void
  ) {
    self.defaults = defaults
    self.loginService = loginService
    self.registerShortcut = registerShortcut
    showMenuBar = defaults.object(forKey: "desktopShowMenuBar") as? Bool ?? true
    shortcut = defaults.string(forKey: "desktopQuestionShortcut") ?? "off"
    if let remembered = defaults.string(forKey: "desktopLastQuestionShortcut"),
      QuestionHotKey.choices.contains(where: { $0.0 == remembered })
    {
      rememberedShortcut = remembered
    } else {
      rememberedShortcut = "control-option-space"
    }
    if QuestionHotKey.choices.contains(where: { $0.0 == shortcut }) {
      rememberedShortcut = shortcut
    }
  }
  public func start() {
    configureShortcut(shortcut)
    refreshLogin()
  }
  public func configureShortcut(_ selection: String) {
    do {
      try registerShortcut(selection) { [weak self] in self?.onQuestion?() }
      shortcut = selection
      defaults.set(selection, forKey: "desktopQuestionShortcut")
      if selection != "off" {
        rememberedShortcut = selection
        defaults.set(selection, forKey: "desktopLastQuestionShortcut")
      }
      shortcutError = nil
    } catch { shortcutError = error.localizedDescription }
  }
  public var shortcutEnabled: Bool { shortcut != "off" }
  public func setShortcutEnabled(_ enabled: Bool) {
    configureShortcut(enabled ? rememberedShortcut : "off")
  }
  public func refreshLogin() {
    let next = loginService.status()
    if next != loginState { loginError = nil }
    loginState = next
  }
  public func setLogin(_ enabled: Bool) async {
    guard !loginBusy else { return }
    loginBusy = true
    defer { loginBusy = false }
    do {
      loginError = nil
      if enabled {
        try loginService.register()
      } else {
        try await loginService.unregister()
      }
      refreshLogin()
      if loginEnabled != enabled {
        loginError = "macOS couldn’t \(enabled ? "enable" : "disable") launch at login. Try again."
      }
    } catch {
      refreshLogin()
      loginError = "Couldn’t change launch at login: \(error.localizedDescription)"
    }
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
    ("control-option-space", "⌃⌥Space"), ("command-shift-space", "⇧⌘Space"),
    ("command-option-j", "⌥⌘J"),
  ]

  func register(_ value: String, action: @escaping () -> Void) throws {
    guard value == "off" || Self.choices.contains(where: { $0.0 == value }) else {
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
