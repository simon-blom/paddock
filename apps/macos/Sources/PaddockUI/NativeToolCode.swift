import AppKit
import SwiftUI

/// ToolCall.vue: content height + 9pt/11pt insets, scrolling only past 260pt.
/// ViewThatFits + maxHeight expands and centers short results in spare space.
/// One TextKit surface keeps selection when a streamed result crosses the cap.
struct NativeToolCode: NSViewRepresentable {
  let text: String
  let textID: String

  func makeNSView(context: Context) -> NativeToolCodeScrollView { NativeToolCodeScrollView() }
  func updateNSView(_ view: NativeToolCodeScrollView, context: Context) {
    view.update(text, id: textID)
  }
  func sizeThatFits(
    _ proposal: ProposedViewSize, nsView: NativeToolCodeScrollView, context: Context
  ) -> CGSize? {
    let width = proposal.width.flatMap { $0.isFinite ? $0 : nil } ?? 760
    return nsView.fittingSize(width: max(1, width))
  }
}

final class NativeToolCodeScrollView: NSScrollView {
  static let maximumHeight: CGFloat = 260
  static let insets = NSSize(width: 11, height: 9)
  let textView = NSTextView(frame: .zero)
  private var measurement: (width: CGFloat, height: CGFloat, scrolls: Bool)?
  private var layingOut = false

  override init(frame: NSRect) {
    super.init(frame: frame)
    borderType = .noBorder
    drawsBackground = false
    hasHorizontalScroller = false
    hasVerticalScroller = false
    automaticallyAdjustsContentInsets = false
    contentInsets = .init()
    textView.isEditable = false
    textView.isSelectable = true
    textView.isRichText = false
    textView.drawsBackground = false
    textView.isVerticallyResizable = false
    textView.isHorizontallyResizable = false
    textView.textContainerInset = Self.insets
    textView.textContainer?.lineFragmentPadding = 0
    textView.textContainer?.widthTracksTextView = false
    textView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
    textView.textColor = .labelColor
    documentView = textView
  }
  required init?(coder: NSCoder) { fatalError("init(coder:) is unavailable") }

  func update(_ text: String, id: String) {
    textView.identifier = .init(id)
    guard textView.string != text else { return }
    let previous = textView.string as NSString
    let selected = textView.selectedRange()
    textView.string = text
    let next = text as NSString
    if NSMaxRange(selected) <= previous.length, NSMaxRange(selected) <= next.length,
      previous.substring(with: selected) == next.substring(with: selected)
    {
      textView.setSelectedRange(selected)
    }
    measurement = nil
    invalidateIntrinsicContentSize()
    needsLayout = true
  }

  func fittingSize(width: CGFloat) -> CGSize {
    CGSize(width: width, height: min(Self.maximumHeight, measure(width: width).height))
  }

  private var gutter: CGFloat {
    NSScroller.scrollerWidth(for: .regular, scrollerStyle: .legacy)
  }
  private func measure(width: CGFloat) -> (height: CGFloat, scrolls: Bool) {
    if let measurement, measurement.width == width {
      return (measurement.height, measurement.scrolls)
    }
    let natural = textHeight(width: width)
    let scrolls = natural > Self.maximumHeight
    let height = scrolls ? textHeight(width: max(1, width - gutter)) : natural
    measurement = (width, height, scrolls)
    return (height, scrolls)
  }
  private func textHeight(width: CGFloat) -> CGFloat {
    guard let container = textView.textContainer, let layout = textView.layoutManager else {
      return 33
    }
    container.containerSize = CGSize(
      width: max(1, width - 2 * Self.insets.width), height: .greatestFiniteMagnitude)
    layout.ensureLayout(for: container)
    return ceil(max(15, layout.usedRect(for: container).height)) + 2 * Self.insets.height
  }

  override func layout() {
    super.layout()
    guard !layingOut else { return }
    layingOut = true
    defer { layingOut = false }
    let size = measure(width: bounds.width)
    if hasVerticalScroller != size.scrolls { hasVerticalScroller = size.scrolls }
    if size.scrolls { PaddockScrollbars.install(on: self) }
    tile()
    let width = max(1, contentSize.width)
    // SwiftUI can probe several widths before placement. Restore TextKit to
    // the actual viewport, never leave the document at the last probe width.
    let height = textHeight(width: width)
    let documentSize = NSSize(width: width, height: height)
    if textView.frame.size != documentSize { textView.setFrameSize(documentSize) }
    let origin = NSPoint(
      x: 0, y: min(max(0, contentView.bounds.minY), max(0, height - contentSize.height)))
    if contentView.bounds.origin != origin { contentView.scroll(to: origin) }
    reflectScrolledClipView(contentView)
  }

  override func scrollWheel(with event: NSEvent) {
    // A short result must not trap the conversation's scrolling gesture.
    if !hasVerticalScroller {
      nextResponder?.scrollWheel(with: event)
    } else {
      super.scrollWheel(with: event)
    }
  }
}
