import AppKit
import SwiftUI

/// Top controls are outside the scrolling viewport. The bottom composer keeps
/// its native floating bar and automatic scroll-edge treatment.
struct StudioConversationBars<Header: View, Footer: View>: ViewModifier {
  @ViewBuilder var header: () -> Header
  @ViewBuilder var footer: () -> Footer

  @ViewBuilder func body(content: Content) -> some View {
    VStack(spacing: 0) {
      header()
      bottomBar(content: content).clipped()
    }
  }

  @ViewBuilder private func bottomBar(content: Content) -> some View {
    if #available(macOS 26.0, *) {
      content
        .safeAreaBar(edge: .bottom, spacing: 0, content: footer)
        .scrollEdgeEffectHidden(true, for: .top)
        .scrollEdgeEffectStyle(.automatic, for: .bottom)
    } else {
      // The package's diagnostic tools still build on macOS 15. Shipping app
      // builds target 26+, where the system provides scroll-edge rendering.
      content
        .safeAreaInset(edge: .bottom, spacing: 0, content: footer)
    }
  }
}

struct StudioConversationChrome<Content: View>: View {
  var title: String? = nil
  var titleColumn: CGRect? = nil
  @ViewBuilder var content: () -> Content
  @Environment(\.workspaceLeadingPaneInset) private var leadingPaneInset
  @State private var titlebarInset: CGFloat = 0

  var body: some View {
    VStack(spacing: 0) {
      if titlebarInset > 0 {
        PaddockStyle.canvas.frame(height: titlebarInset)
          .contentShape(Rectangle()).gesture(WindowDragGesture())
          .overlay {
            GeometryReader { geometry in
              let clearance = leadingPaneInset > 0 ? CGFloat(240) : 20
              let column =
                titleColumn
                ?? StudioColumnLayout.resolve(
                  available: geometry.size.width, viewport: nil)
              if let title, !title.isEmpty,
                let frame = StudioConversationTitleLayout.frame(
                  available: geometry.size.width, height: titlebarInset,
                  column: column, controlsClearance: clearance)
              {
                Text(title)
                  .font(.system(size: 13, weight: .medium))
                  .lineLimit(1).truncationMode(.tail)
                  .help(title).accessibilityLabel("Conversation: \(title)")
                  .accessibilityIdentifier("conversation-header-title")
                  .frame(width: frame.width, height: frame.height, alignment: .leading)
                  .offset(x: frame.minX)
              }
            }
          }
          .overlay(alignment: .bottom) { Divider() }
      }
      content().frame(maxWidth: .infinity, maxHeight: .infinity).clipped()
    }
    .background(PaddockStyle.canvas)
    .background(alignment: .top) {
      StudioWindowChromeProbe { titlebarInset = $0 }
        .frame(height: 0).allowsHitTesting(false).accessibilityHidden(true)
    }
  }
}

enum StudioConversationTitleLayout {
  /// Anchor to the chat column, never to the center of the whole window or
  /// the center of an asymmetrically padded toolbar. Keep native controls clear.
  static func frame(available: CGFloat, height: CGFloat, column: CGRect, controlsClearance: CGFloat)
    -> CGRect?
  {
    guard available.isFinite, height.isFinite, column.minX.isFinite, column.maxX.isFinite,
      controlsClearance.isFinite, height > 0
    else { return nil }
    let left = max(0, max(column.minX, controlsClearance))
    let right = min(column.maxX, available - 52)
    guard right - left >= 100 else { return nil }
    return CGRect(x: left, y: 0, width: right - left, height: height)
  }
}

private struct StudioWindowChromeProbe: NSViewRepresentable {
  var onInsetChange: (CGFloat) -> Void
  func makeNSView(context: Context) -> StudioWindowChromeProbeView { StudioWindowChromeProbeView() }
  func updateNSView(_ view: StudioWindowChromeProbeView, context: Context) {
    view.onInsetChange = onInsetChange
    view.refreshClearance()
  }
}

/// The nested split hosts deliberately do not re-inherit window safe areas.
/// Measure the actual titlebar overlap for this pane's non-scrolling header.
/// This zero-height probe draws nothing and never observes scroll offsets.
final class StudioWindowChromeProbeView: NSView {
  var onInsetChange: ((CGFloat) -> Void)?
  private var windowObservation: NSKeyValueObservation?
  private var publishedInset: CGFloat = -1
  private var measuredInset: CGFloat = 0
  private var updateQueued = false

  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    windowObservation = window?.observe(\.contentLayoutRect, options: [.new]) { [weak self] _, _ in
      Task { @MainActor [weak self] in self?.refreshClearance() }
    }
    refreshClearance()
  }

  override func layout() {
    super.layout()
    refreshClearance()
  }

  func refreshClearance() {
    measuredInset =
      window.map { max(0, convert(bounds, to: nil).maxY - $0.contentLayoutRect.maxY) } ?? 0
    guard measuredInset != publishedInset, !updateQueued else { return }
    updateQueued = true
    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      updateQueued = false
      guard measuredInset != publishedInset else { return }
      publishedInset = measuredInset
      onInsetChange?(measuredInset)
    }
  }
}
