import AppKit
import SwiftUI

/// AppKit owns scrolling, dragging, keyboard/AX actions and thumb proportion.
/// Only the painting changes; the full native hit target remains intact.
public final class PaddockScroller: NSScroller {
  public override class var isCompatibleWithOverlayScrollers: Bool { false }
  public override var isOpaque: Bool { false }

  public static func thumb(in slot: NSRect, horizontal: Bool = false) -> NSRect {
    if horizontal {
      return NSRect(x: slot.minX, y: slot.midY - 2, width: slot.width, height: 4)
    }
    return NSRect(x: slot.midX - 2, y: slot.minY, width: 4, height: slot.height)
  }
  public override func drawKnob() {
    guard knobProportion < 1 else { return }
    let thumb = Self.thumb(in: rect(for: .knob), horizontal: bounds.width > bounds.height)
    let contrast = NSWorkspace.shared.accessibilityDisplayShouldIncreaseContrast
    NSColor.labelColor.withAlphaComponent(contrast ? 0.65 : 0.28).setFill()
    NSBezierPath(roundedRect: thumb, xRadius: 2, yRadius: 2).fill()
  }
  public override func drawKnobSlot(in slotRect: NSRect, highlight flag: Bool) {}
}

@MainActor public enum PaddockScrollbars {
  public static func install(on scroll: NSScrollView) {
    // Never enable an axis the owner deliberately hid (attachment/tab strips).
    if scroll.hasVerticalScroller, !(scroll.verticalScroller is PaddockScroller) {
      // Keep the standard layout/hit width: SwiftUI budgets that same width.
      // A small control creates a four-point disagreement with its layout.
      let scroller = PaddockScroller()
      scroller.controlSize = .regular
      scroll.verticalScroller = scroller
    }
    if scroll.hasHorizontalScroller, !(scroll.horizontalScroller is PaddockScroller) {
      let scroller = PaddockScroller()
      scroller.controlSize = .regular
      scroll.horizontalScroller = scroller
    }
    if scroll.scrollerStyle != .legacy { scroll.scrollerStyle = .legacy }
    if !scroll.autohidesScrollers { scroll.autohidesScrollers = true }
  }

  /// For an explicitly owned, small hosting boundary (TextEditor/Markdown).
  /// No app-wide window scanning, timers, swizzling or preference overrides.
  public static func install(in root: NSView) {
    if let scroll = root as? NSScrollView { install(on: scroll) }
    for child in root.subviews { install(in: child) }
  }
}

/// Lives inside the scroll content, so it can only affect its enclosing view.
public struct PaddockScrollStyle: NSViewRepresentable {
  var gutterChanged: ((CGFloat) -> Void)?
  public init(gutterChanged: ((CGFloat) -> Void)? = nil) { self.gutterChanged = gutterChanged }
  public func makeNSView(context: Context) -> Installer { Installer() }
  public func updateNSView(_ view: Installer, context: Context) {
    view.gutterChanged = gutterChanged
    view.install()
  }
  public final class Installer: NSView {
    var gutterChanged: ((CGFloat) -> Void)?
    private var lastGutter: CGFloat?
    private weak var observedClip: NSClipView?
    private var measurementQueued = false
    public override func viewDidMoveToWindow() {
      super.viewDidMoveToWindow()
      if window == nil {
        NotificationCenter.default.removeObserver(self)
        observedClip = nil
      } else {
        install()
      }
    }
    public override func layout() {
      super.layout()
      install()
    }
    isolated deinit { NotificationCenter.default.removeObserver(self) }
    func install() {
      guard let scroll = enclosingScrollView else { return }
      PaddockScrollbars.install(on: scroll)
      guard gutterChanged != nil else { return }
      if observedClip !== scroll.contentView {
        NotificationCenter.default.removeObserver(self)
        observedClip = scroll.contentView
        scroll.contentView.postsFrameChangedNotifications = true
        NotificationCenter.default.addObserver(
          self, selector: #selector(measureGutter),
          name: NSView.frameDidChangeNotification, object: scroll.contentView)
      }
      measureGutter()
    }
    @objc private func measureGutter() {
      guard !measurementQueued, let scroll = enclosingScrollView,
        lastGutter != reservedGutter(scroll)
      else { return }
      measurementQueued = true
      Task { @MainActor [weak self] in
        // NSScrollView must finish tiling after its control size changes. Its
        // old/default gutter is not the width the content will actually see.
        await Task.yield()
        guard let self else { return }
        self.measurementQueued = false
        guard let scroll = self.enclosingScrollView else { return }
        let gutter = self.reservedGutter(scroll)
        guard self.lastGutter != gutter else { return }
        self.lastGutter = gutter
        self.gutterChanged?(gutter)
      }
    }
    private func reservedGutter(_ scroll: NSScrollView) -> CGFloat {
      scroll.verticalScroller?.isHidden == false
        ? max(0, scroll.bounds.width - scroll.contentView.frame.width) : 0
    }
  }
}

/// One style for Studio, Manager, popovers and nested code/diagram scrolling.
/// A centered transcript compensates the native scrollbar gutter so it remains
/// aligned with its floating composer, which is outside the scroll view.
public struct PaddockScrollView<Content: View>: View {
  let axes: Axis.Set
  let centersContent: Bool
  let content: Content
  @State private var gutter: CGFloat = 0
  public init(
    _ axes: Axis.Set = .vertical, centersContent: Bool = false,
    @ViewBuilder content: () -> Content
  ) {
    self.axes = axes
    self.centersContent = centersContent
    self.content = content()
  }
  public var body: some View {
    ScrollView(axes) {
      content.padding(.leading, centersContent ? gutter : 0)
        .background(
          PaddockScrollStyle(
            gutterChanged: centersContent ? { if gutter != $0 { gutter = $0 } } : nil))
    }
  }
}

/// Keep TextEditor's binding, native undo and focus; style only its own host.
public struct PaddockTextEditor: NSViewRepresentable {
  @Binding var text: String
  public init(text: Binding<String>) { _text = text }
  public func makeNSView(context: Context) -> PaddockScrollHostingView {
    PaddockScrollHostingView(
      rootView: AnyView(TextEditor(text: $text).environment(\.self, context.environment)))
  }
  public func updateNSView(_ view: PaddockScrollHostingView, context: Context) {
    view.rootView = AnyView(TextEditor(text: $text).environment(\.self, context.environment))
  }
  public func sizeThatFits(
    _ proposal: ProposedViewSize, nsView: PaddockScrollHostingView, context: Context
  ) -> CGSize? {
    guard let width = proposal.width, let height = proposal.height else { return nil }
    return CGSize(width: width, height: height)
  }
}

open class PaddockScrollHostingView: NSHostingView<AnyView> {
  open override func layout() {
    super.layout()
    PaddockScrollbars.install(in: self)
  }
}
