import AppKit
import SwiftUI

/// One selectable text surface, not one SwiftUI Text per log row. Append and
/// prefix eviction update text storage in place, preserving retained selection.
struct EndpointLogText: NSViewRepresentable {
  let lines: [EndpointLogLine]
  let jump: Int
  @Binding var following: Bool
  func makeNSView(context: Context) -> LogScrollView {
    let view = LogScrollView()
    view.onFollowing = { value in
      Task { @MainActor in following = value }
    }
    return view
  }
  func updateNSView(_ view: LogScrollView, context: Context) {
    view.onFollowing = { value in Task { @MainActor in following = value } }
    view.update(lines, jump: jump)
  }
}

final class LogScrollView: NSScrollView, NSTextViewDelegate {
  let text = NSTextView(frame: .zero)
  var onFollowing: ((Bool) -> Void)?
  private(set) var following = true
  private var rows: [EndpointLogLine] = []
  private var jump = 0
  private var updating = false
  private var observation: NSObjectProtocol?
  override init(frame: NSRect) {
    super.init(frame: frame)
    borderType = .noBorder
    drawsBackground = false
    hasVerticalScroller = true
    autohidesScrollers = true
    PaddockScrollbars.install(on: self)
    text.isEditable = false
    text.isSelectable = true
    text.isRichText = false
    text.drawsBackground = false
    text.isVerticallyResizable = true
    text.isHorizontallyResizable = false
    text.autoresizingMask = [.width]
    text.minSize = .zero
    text.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
    text.textContainer?.widthTracksTextView = true
    text.textContainer?.containerSize = NSSize(width: 600, height: CGFloat.greatestFiniteMagnitude)
    text.textContainerInset = NSSize(width: 10, height: 10)
    text.delegate = self
    text.setAccessibilityLabel("Model log lines")
    documentView = text
    text.frame = NSRect(origin: .zero, size: contentSize)
    contentView.postsBoundsChangedNotifications = true
    observation = NotificationCenter.default.addObserver(
      forName: NSView.boundsDidChangeNotification, object: contentView, queue: .main
    ) { [weak self] _ in
      MainActor.assumeIsolated { self?.scrolled() }
    }
  }
  required init?(coder: NSCoder) { fatalError("init(coder:) is unavailable") }
  isolated deinit { if let observation { NotificationCenter.default.removeObserver(observation) } }
  func textViewDidChangeSelection(_ notification: Notification) {
    if !updating && text.selectedRange().length > 0 { setFollowing(false) }
  }
  private func setFollowing(_ value: Bool) {
    guard value != following else { return }
    following = value
    onFollowing?(value)
  }
  private func scrolled() {
    guard !updating else { return }
    setFollowing(
      contentView.bounds.maxY >= text.bounds.height - 6 && text.selectedRange().length == 0)
  }
  func update(_ next: [EndpointLogLine], jump requested: Int) {
    let jumping = requested != jump
    jump = requested
    guard next != rows || jumping else { return }
    updating = true
    defer { updating = false }
    if jumping {
      text.setSelectedRange(NSRange(location: text.string.utf16.count, length: 0))
      setFollowing(true)
    }
    let selected = text.selectedRange()
    let origin = contentView.bounds.origin
    let drop = next.first.flatMap { first in rows.firstIndex { $0.id == first.id } }
    let overlap = drop.map { min(rows.count - $0, next.count) } ?? 0
    let incremental =
      drop.map { Array(rows[$0..<($0 + overlap)]) == Array(next.prefix(overlap)) } ?? rows.isEmpty
    let removed =
      incremental
      ? rows.prefix(drop ?? 0).reduce(0) { $0 + $1.display.utf16.count } : text.string.utf16.count
    var removedHeight: CGFloat = 0
    if removed > 0, incremental, let layout = text.layoutManager, let container = text.textContainer
    {
      layout.ensureLayout(for: container)
      let glyphs = layout.glyphRange(
        forCharacterRange: NSRange(location: 0, length: removed), actualCharacterRange: nil)
      removedHeight = layout.boundingRect(forGlyphRange: glyphs, in: container).height
    }
    guard let storage = text.textStorage else { return }
    storage.beginEditing()
    if removed > 0 { storage.deleteCharacters(in: NSRange(location: 0, length: removed)) }
    for row in next.dropFirst(incremental ? overlap : 0) {
      var attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.monospacedSystemFont(ofSize: 11, weight: .regular),
        .foregroundColor: NSColor.labelColor,
      ]
      if let module = row.module { attributes[.toolTip] = module }
      let rendered = NSMutableAttributedString(string: row.display, attributes: attributes)
      if let level = row.level, let range = row.display.range(of: level) {
        let color: NSColor =
          level == "ERROR" ? .systemRed : level == "WARN" ? .systemOrange : .secondaryLabelColor
        rendered.addAttribute(
          .foregroundColor, value: color, range: NSRange(range, in: row.display))
      }
      storage.append(rendered)
    }
    storage.endEditing()
    if incremental && selected.location >= removed
      && NSMaxRange(selected) - removed <= storage.length
    {
      text.setSelectedRange(NSRange(location: selected.location - removed, length: selected.length))
    } else {
      text.setSelectedRange(NSRange(location: 0, length: 0))
    }
    rows = next
    if let container = text.textContainer { text.layoutManager?.ensureLayout(for: container) }
    if following {
      text.scrollRangeToVisible(NSRange(location: storage.length, length: 0))
    } else {
      contentView.scroll(to: NSPoint(x: origin.x, y: max(0, origin.y - removedHeight)))
      reflectScrolledClipView(contentView)
    }
  }
}
