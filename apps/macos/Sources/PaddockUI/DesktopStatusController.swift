import AppKit
import SwiftUI

/// Own the status item's public button so presentation has a real screen
/// anchor. The menu itself reuses the existing SwiftUI inventory/actions.
/// No private MenuBarExtra view traversal or second menu implementation.
@MainActor public final class DesktopStatusController {
  private(set) var item: NSStatusItem?
  private var menu: NSHostingMenu<DesktopMenu>?
  private let workspace: WorkspaceModel
  private let open: (DesktopAction) -> Void
  private let question: (NSRect?) -> Void
  private let settings: () -> Void

  public init(
    workspace: WorkspaceModel, open: @escaping (DesktopAction) -> Void,
    question: @escaping (NSRect?) -> Void, settings: @escaping () -> Void
  ) {
    self.workspace = workspace
    self.open = open
    self.question = question
    self.settings = settings
  }
  public var anchor: NSRect? {
    guard let button = item?.button, let window = button.window else { return nil }
    return window.convertToScreen(button.convert(button.bounds, to: nil))
  }
  public func setVisible(_ visible: Bool) {
    if !visible {
      if let item { NSStatusBar.system.removeStatusItem(item) }
      item = nil
      menu = nil
      return
    }
    guard item == nil else { return }
    let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
    item.autosaveName = "PaddockModels"
    item.button?.image = Self.menuBarImage
    item.button?.image?.isTemplate = true
    item.button?.setAccessibilityLabel("Paddock models")
    item.button?.setAccessibilityIdentifier("paddock-status-item")
    let menu = NSHostingMenu(
      rootView: DesktopMenu(
        workspace: workspace, open: open,
        question: { [weak self] in
          guard let self else { return }
          let anchor = self.anchor
          // Finish the system menu's tracking/dismissal before showing a panel.
          DispatchQueue.main.async { [weak self] in self?.question(anchor) }
        }, settings: settings))
    item.menu = menu
    self.item = item
    self.menu = menu
  }
  /// The Paddock mark drawn for the menu bar (PaddockMenuBar.svg, 18 pt, black
  /// only), used as a template so macOS tints it. The system symbol stands in
  /// only if the resource bundle is missing.
  static let menuBarImage: NSImage? = {
    let image =
      ProviderArtwork.resources?.url(forResource: "PaddockMenuBar", withExtension: "svg")
      .flatMap(NSImage.init(contentsOf:))
      ?? NSImage(systemSymbolName: "square.stack.3d.up", accessibilityDescription: nil)
    image?.accessibilityDescription = "Paddock models"
    return image
  }()

  isolated deinit { if let item { NSStatusBar.system.removeStatusItem(item) } }
}
