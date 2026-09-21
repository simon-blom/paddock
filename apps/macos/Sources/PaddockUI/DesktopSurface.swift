import AppKit
import SwiftUI

extension View {
  /// Apply at a top-level app-owned window, not a nested popover. Content paint
  /// and NSWindow chrome are independent; both must use our opaque palette.
  func desktopWindowSurface(appearance: WorkspaceAppearance) -> some View {
    background {
      PaddockStyle.canvas.ignoresSafeArea()
        .background(
          DesktopWindowBackdrop(appearance: appearance).allowsHitTesting(false)
            .accessibilityHidden(true))
    }
    .containerBackground(PaddockStyle.canvas, for: .window)
    .tint(PaddockStyle.accent)
  }
}

private struct DesktopWindowBackdrop: NSViewRepresentable {
  let appearance: WorkspaceAppearance
  func makeNSView(context: Context) -> DesktopWindowBackdropView { DesktopWindowBackdropView() }
  func updateNSView(_ view: DesktopWindowBackdropView, context: Context) {
    view.requestedAppearance = appearance
    view.scheduleSurface()
  }
}

final class DesktopWindowBackdropView: NSView {
  var requestedAppearance: WorkspaceAppearance = .system
  private var updatePending = false
  override func hitTest(_ point: NSPoint) -> NSView? { nil }
  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    scheduleSurface()
  }
  func scheduleSurface() {
    guard !updatePending else { return }
    updatePending = true
    // Finish attaching the hosting controller before changing safe-area chrome.
    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      self.updatePending = false
      guard let window = self.window else { return }
      // NSHostingController can retain the last explicit appearance when the
      // SwiftUI preference changes back to nil. Clear it for System mode.
      let appearance: NSAppearance? =
        switch self.requestedAppearance {
        case .system: nil
        case .light: NSAppearance(named: .aqua)
        case .dark: NSAppearance(named: .darkAqua)
        }
      if window.appearance?.name != appearance?.name { window.appearance = appearance }
      window.backgroundColor = PaddockStyle.nsColor("canvas")
      window.isOpaque = true
      if !window.styleMask.contains(.fullSizeContentView) {
        let frame = window.frame
        window.styleMask.insert(.fullSizeContentView)
        window.setFrame(frame, display: false)
      }
      window.titlebarAppearsTransparent = true
    }
  }
}

/// App-owned confirmations keep NSAlert's keyboard, cancellation and semantic
/// controls. Only its wallpaper-tinted window background is replaced.
public enum DesktopAlert {
  @MainActor public static func make() -> NSAlert {
    let alert = NSAlert()
    let appearance =
      WorkspaceAppearance(
        rawValue: UserDefaults.standard.string(forKey: "workspaceAppearance") ?? "System")
      ?? .system
    switch appearance {
    case .system: break
    case .light: alert.window.appearance = NSAppearance(named: .aqua)
    case .dark: alert.window.appearance = NSAppearance(named: .darkAqua)
    }
    alert.window.backgroundColor = PaddockStyle.nsColor("canvas")
    alert.window.isOpaque = true
    if let content = alert.window.contentView {
      // NSAlert's own material is above the NSWindow background. Paint below
      // its controls using a public content subview, not a private glass view.
      let backdrop = DesktopOpaqueBackdrop(frame: content.bounds)
      backdrop.autoresizingMask = [.width, .height]
      content.addSubview(backdrop, positioned: .below, relativeTo: nil)
    }
    return alert
  }
}

private final class DesktopOpaqueBackdrop: NSView {
  override var isOpaque: Bool { true }
  override func hitTest(_ point: NSPoint) -> NSView? { nil }
  override func draw(_ dirtyRect: NSRect) {
    PaddockStyle.nsColor("canvas").setFill()
    dirtyRect.fill()
  }
  override func viewDidChangeEffectiveAppearance() {
    super.viewDidChangeEffectiveAppearance()
    needsDisplay = true
  }
}
