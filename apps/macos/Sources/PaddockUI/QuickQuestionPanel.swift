import AppKit
import QuartzCore
import SwiftUI

public final class QuickQuestionPanel: NSPanel {
  public override var canBecomeKey: Bool { true }
  public override var canBecomeMain: Bool { false }
  var dismiss: (() -> Void)?
  public override func cancelOperation(_ sender: Any?) { dismiss?() }
  public override func performClose(_ sender: Any?) { dismiss?() }
}

enum QuickPanelLayout {
  static let width: CGFloat = 460
  static func frame(height: CGFloat, visible: NSRect, anchor: NSRect?) -> NSRect {
    let width = min(Self.width, max(0, visible.width - 24))
    let height = min(max(132, height), max(0, visible.height - 24))
    let x = min(
      visible.maxX - width - 12, max(visible.minX + 12, (anchor?.midX ?? visible.midX) - width / 2))
    let top = min(visible.maxY - 8, anchor.map { $0.minY - 8 } ?? visible.maxY - 8)
    return NSRect(x: x, y: max(visible.minY + 12, top - height), width: width, height: height)
  }
  static func duration(showing: Bool, reduceMotion: Bool) -> TimeInterval {
    reduceMotion ? 0 : showing ? 0.18 : 0.12
  }
}

/// A retained, nonactivating input surface, not a second app window. Only its
/// rounded corners are transparent; the content has no wallpaper/material tint.
@MainActor public final class QuickQuestionController: NSObject, NSWindowDelegate {
  public let draft = QuickQuestionModel()
  private(set) var panel: QuickQuestionPanel?
  private(set) var presented = false
  private var previousApp: NSRunningApplication?
  private var anchor: NSRect?
  private var screen: NSScreen?
  private var target = NSRect.zero
  private var transition: UInt64 = 0
  private var animating = false
  private var pendingHeight: CGFloat?
  private var localClick: Any?
  private var globalClick: Any?
  private var observations: [NSObjectProtocol] = []
  // Tests can exercise the no-motion path without changing OS preferences.
  var reduceMotion: () -> Bool = { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }

  public override init() { super.init() }
  public func show(
    workspace: WorkspaceModel, anchor: NSRect? = nil, open: @escaping (DesktopAction) -> Void
  ) {
    if presented {
      panel?.makeKey()
      return
    }
    previousApp = NSWorkspace.shared.frontmostApplication
    self.anchor = anchor
    screen =
      NSScreen.screens.first { screen in
        anchor.map { screen.frame.intersects($0) } ?? screen.frame.contains(NSEvent.mouseLocation)
      } ?? NSScreen.main
    if panel == nil {
      let panel = QuickQuestionPanel(
        contentRect: NSRect(x: 0, y: 0, width: QuickPanelLayout.width, height: 148),
        styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
      panel.title = "New Question"  // Accessibility/window identity, never drawn.
      panel.identifier = NSUserInterfaceItemIdentifier("quick-question")
      panel.backgroundColor = .clear
      panel.isOpaque = false
      panel.hasShadow = true
      panel.isReleasedWhenClosed = false
      panel.isFloatingPanel = true
      panel.level = .floating
      panel.hidesOnDeactivate = false
      panel.animationBehavior = .none
      panel.collectionBehavior = [
        .moveToActiveSpace, .fullScreenAuxiliary, .transient, .ignoresCycle,
      ]
      panel.delegate = self
      panel.dismiss = { [weak self] in self?.hide() }
      let host = NSHostingController(
        rootView: QuickQuestionView(
          draft: draft, workspace: workspace,
          openStudio: { [weak self] in
            self?.hide(restoreFocus: false, animated: false)
            open(.studio)
          },
          openManager: { [weak self] in
            self?.hide(restoreFocus: false, animated: false)
            open(.manager)
          },
          close: { [weak self] in self?.hide() },
          resize: { [weak self] in self?.resize($0) },
          appearanceChanged: { [weak panel] appearance in
            panel?.appearance =
              appearance == .system
              ? nil : NSAppearance(named: appearance == .dark ? .darkAqua : .aqua)
          }))
      host.sizingOptions = []
      panel.contentViewController = host
      self.panel = panel
    }
    guard let panel, let screen else { return }
    presented = true
    transition &+= 1
    let ticket = transition
    target = QuickPanelLayout.frame(
      height: pendingHeight ?? panel.frame.height, visible: screen.visibleFrame, anchor: anchor)
    let duration = QuickPanelLayout.duration(showing: true, reduceMotion: reduceMotion())
    animating = duration > 0
    panel.alphaValue = duration > 0 ? 0 : 1
    panel.setFrame(duration > 0 ? target.offsetBy(dx: 0, dy: 12) : target, display: false)
    panel.makeKeyAndOrderFront(nil)
    panel.contentView?.layoutSubtreeIfNeeded()
    focusEditor()
    monitorDismissal()
    if duration > 0 {
      NSAnimationContext.runAnimationGroup { context in
        context.duration = duration
        context.timingFunction = CAMediaTimingFunction(name: .easeOut)
        panel.animator().alphaValue = 1
        panel.animator().setFrame(target, display: true)
      } completionHandler: { [weak self] in
        MainActor.assumeIsolated {
          guard let self, self.transition == ticket else { return }
          self.animating = false
          if let height = self.pendingHeight { self.resize(height) }
        }
      }
    }
  }
  private func focusEditor() {
    guard let panel, let root = panel.contentView else { return }
    func editor(_ view: NSView) -> DraftTextView? {
      if let input = view as? DraftTextView { return input }
      return view.subviews.lazy.compactMap(editor).first
    }
    if let input = editor(root) { panel.makeFirstResponder(input) }
  }
  private func resize(_ height: CGFloat) {
    guard height.isFinite else { return }
    pendingHeight = height
    guard presented, !animating, let panel, let screen else { return }
    let frame = QuickPanelLayout.frame(
      height: ceil(height), visible: screen.visibleFrame, anchor: anchor)
    guard frame != target else { return }
    target = frame
    panel.setFrame(frame, display: true)  // Grow downwards; keep the caret stationary.
    panel.invalidateShadow()
  }
  public func hide(restoreFocus: Bool = true, animated: Bool = true) {
    guard presented, let panel else { return }
    presented = false
    transition &+= 1
    let ticket = transition
    stopMonitoring()
    let duration =
      animated ? QuickPanelLayout.duration(showing: false, reduceMotion: reduceMotion()) : 0
    animating = duration > 0
    let finish: @MainActor @Sendable () -> Void = { [weak self, weak panel] in
      guard let self, transition == ticket else { return }
      panel?.orderOut(nil)
      panel?.alphaValue = 1
      animating = false
      if restoreFocus, let previousApp,
        previousApp.processIdentifier != ProcessInfo.processInfo.processIdentifier
      {
        previousApp.activate(options: [])
      }
    }
    if duration == 0 {
      finish()
      return
    }
    NSAnimationContext.runAnimationGroup { context in
      context.duration = duration
      context.timingFunction = CAMediaTimingFunction(name: .easeIn)
      panel.animator().alphaValue = 0
      panel.animator().setFrame(target.offsetBy(dx: 0, dy: 8), display: true)
    } completionHandler: {
      MainActor.assumeIsolated { finish() }
    }
  }
  private func monitorDismissal() {
    stopMonitoring()
    localClick = NSEvent.addLocalMonitorForEvents(matching: [
      .leftMouseDown, .rightMouseDown, .otherMouseDown,
    ]) { [weak self] event in
      self?.clicked(window: event.window)
      return event  // Never swallow the click meant for another application/control.
    }
    globalClick = NSEvent.addGlobalMonitorForEvents(matching: [
      .leftMouseDown, .rightMouseDown, .otherMouseDown,
    ]) { [weak self] _ in
      guard self?.panel?.attachedSheet == nil else { return }
      self?.hide(restoreFocus: false)
    }
    for (center, name) in [
      (NotificationCenter.default, NSApplication.didChangeScreenParametersNotification),
      (NSWorkspace.shared.notificationCenter, NSWorkspace.activeSpaceDidChangeNotification),
      (NSWorkspace.shared.notificationCenter, NSWorkspace.didActivateApplicationNotification),
    ] {
      observations.append(
        center.addObserver(forName: name, object: nil, queue: .main) { [weak self] notification in
          let activatedPID =
            (notification.userInfo?[NSWorkspace.applicationUserInfoKey]
            as? NSRunningApplication)?.processIdentifier
          MainActor.assumeIsolated {
            if name == NSWorkspace.didActivateApplicationNotification,
              activatedPID == ProcessInfo.processInfo.processIdentifier
            {
              return
            }
            guard self?.panel?.attachedSheet == nil else { return }
            self?.hide(restoreFocus: false, animated: false)
          }
        })
    }
  }
  func clicked(window: NSWindow?) {
    guard let panel, panel.attachedSheet == nil else { return }
    // Owned sheets and pop-up menus stay interactive. A click in another app
    // is handled by the global mouse-only monitor; no keyboard monitoring.
    if window === panel || window?.sheetParent === panel
      || panel.childWindows?.contains(where: { $0 === window }) == true
    {
      return
    }
    if window?.level.rawValue ?? 0 >= NSWindow.Level.popUpMenu.rawValue { return }
    hide(restoreFocus: false)
  }
  private func stopMonitoring() {
    if let localClick { NSEvent.removeMonitor(localClick) }
    if let globalClick { NSEvent.removeMonitor(globalClick) }
    localClick = nil
    globalClick = nil
    for token in observations {
      NotificationCenter.default.removeObserver(token)
      NSWorkspace.shared.notificationCenter.removeObserver(token)
    }
    observations.removeAll()
  }
  public func windowShouldClose(_ sender: NSWindow) -> Bool {
    hide()
    return false
  }
  isolated deinit {
    stopMonitoring()
    panel?.orderOut(nil)
  }
}
