import AppKit
import SwiftUI

/// Keep system positioning, dismissal and keyboard ownership. Only the
/// presentation surface and controls are ours; no private popover/window hacks.
extension View {
  func studioPopoverSurface() -> some View {
    font(.system(size: 12))
      .controlSize(.small)
      .tint(PaddockStyle.accent)
      // Fill the entire popover, including its system insets and arrow, not
      // just the rectangular content area. Controls still respect safe areas.
      .background {
        PaddockStyle.popup.ignoresSafeArea()
          .background(StudioPopoverExtent().allowsHitTesting(false).accessibilityHidden(true))
      }
      .presentationBackground(PaddockStyle.popup)
  }
}

/// SwiftUI's macOS popover can leave NSPopover.hasFullSizeContent false. In
/// that mode the hosting view ends *inside* the popover: ignoresSafeArea alone
/// cannot paint the exposed material. Configure the enclosing NSPopover via
/// its public responder chain and public full-size-content API (macOS 14+).
/// No private view names, KVC, window mutation or replacement presenter.
private struct StudioPopoverExtent: NSViewRepresentable {
  func makeNSView(context: Context) -> StudioPopoverExtentView { StudioPopoverExtentView() }
  func updateNSView(_ view: StudioPopoverExtentView, context: Context) { view.scheduleUpdate() }
}

final class StudioPopoverExtentView: NSView {
  private var updatePending = false
  override func hitTest(_ point: NSPoint) -> NSView? { nil }
  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    scheduleUpdate()
  }
  override func viewDidMoveToSuperview() {
    super.viewDidMoveToSuperview()
    scheduleUpdate()
  }
  override func layout() {
    super.layout()
    // SwiftUI can attach the hosting controller after viewDidMoveToWindow's
    // deferred callback. Retry on actual layout, not a timer or polling loop.
    scheduleUpdate()
  }
  func scheduleUpdate() {
    guard !updatePending else { return }
    updatePending = true
    // The hosting controller joins the responder chain during presentation.
    // Wait one run-loop turn; do not poll or keep a popover/window alive.
    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      self.updatePending = false
      guard self.window != nil else { return }
      var responder: NSResponder? = self
      var seen = Set<ObjectIdentifier>()
      while let current = responder, seen.insert(ObjectIdentifier(current)).inserted {
        if let popover = current as? NSPopover {
          if !popover.hasFullSizeContent { popover.hasFullSizeContent = true }
          return
        }
        responder = current.nextResponder
      }
    }
  }
}

struct StudioPopoverHeading: View {
  let title: String
  var subtitle: String? = nil
  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(title).font(.system(size: 13, weight: .semibold)).lineLimit(2)
      if let subtitle {
        Text(subtitle).font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(2)
      }
    }.frame(maxWidth: .infinity, alignment: .leading)
  }
}

struct StudioPopoverChoice: View {
  let title: String
  var subtitle: String? = nil
  let selected: Bool
  let action: () -> Void
  var body: some View {
    Button(action: action) {
      HStack(spacing: 12) {
        VStack(alignment: .leading, spacing: 3) {
          Text(title).font(.system(size: 12, weight: .medium))
          if let subtitle {
            Text(subtitle).font(.system(size: 11)).foregroundStyle(.secondary)
              .lineLimit(2).fixedSize(horizontal: false, vertical: true)
          }
        }
        Spacer(minLength: 0)
        Image(systemName: "checkmark").font(.system(size: 11, weight: .semibold))
          .frame(width: 14).opacity(selected ? 1 : 0).accessibilityHidden(true)
      }.padding(.horizontal, 9).padding(.vertical, 8)
        .frame(maxWidth: .infinity, minHeight: 32, alignment: .leading)
        .contentShape(Rectangle())
    }.buttonStyle(QuietButtonStyle())
      .accessibilityAddTraits(selected ? .isSelected : [])
  }
}

struct StudioPopoverFieldStyle: TextFieldStyle {
  func _body(configuration: TextField<Self._Label>) -> some View {
    configuration.textFieldStyle(.plain).font(.system(size: 12)).padding(.horizontal, 9)
      .frame(height: 30)
      .background(
        PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
      )
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.control).strokeBorder(
          PaddockStyle.border))
  }
}
