import AppKit

/// Preserve the passage under the top of the viewport as native Markdown
/// reflows beside a document. A raw scroll offset points at different text
/// after line wrapping. This is scoped to our native transcript's TextKit 1
/// surfaces, never the TextKit 2 composer or WebKit's private view hierarchy.
@MainActor struct StudioReadingAnchor {
  weak var scroll: NSScrollView?
  weak var text: NSTextView?
  let character: Int
  let lineOffset: CGFloat

  static func capture(in root: NSView) -> Self? {
    for scroll in views(NSScrollView.self, in: root) {
      guard let document = scroll.documentView,
        document.bounds.height > scroll.documentVisibleRect.height
          - scroll.contentInsets.top - scroll.contentInsets.bottom + 4,
        document.bounds.maxY - scroll.documentVisibleRect.maxY + scroll.contentInsets.bottom > 4
      else { continue }  // Follow-tail remains owned by the transcript.
      let top = scroll.documentVisibleRect.minY + scroll.contentInsets.top
      for text in views(NSTextView.self, in: document) where !(text is DraftTextView) {
        let frame = text.convert(text.bounds, to: document)
        guard frame.maxY > top, !text.string.isEmpty,
          let manager = text.layoutManager, let container = text.textContainer
        else { continue }
        let point = text.convert(NSPoint(x: frame.minX, y: max(top, frame.minY)), from: document)
        let index = manager.characterIndex(
          for: NSPoint(x: 0, y: max(0, point.y - text.textContainerOrigin.y)),
          in: container, fractionOfDistanceBetweenInsertionPoints: nil)
        let character = min(index, (text.string as NSString).length - 1)
        guard let y = lineY(text, character: character, document: document) else { continue }
        return Self(scroll: scroll, text: text, character: character, lineOffset: top - y)
      }
    }
    return nil
  }

  func restore() {
    guard let scroll, let text, let document = scroll.documentView,
      let y = Self.lineY(text, character: character, document: document)
    else { return }
    let minimum = -scroll.contentInsets.top
    let maximum = max(
      minimum,
      document.bounds.height + scroll.contentInsets.bottom - scroll.contentView.bounds.height)
    let origin = NSPoint(
      x: scroll.contentView.bounds.minX,
      y: min(maximum, max(minimum, y + lineOffset - scroll.contentInsets.top)))
    scroll.contentView.scroll(to: origin)
    scroll.reflectScrolledClipView(scroll.contentView)
  }

  private static func lineY(_ text: NSTextView, character: Int, document: NSView) -> CGFloat? {
    guard character < (text.string as NSString).length,
      let manager = text.layoutManager, text.textContainer != nil
    else { return nil }
    let glyph = manager.glyphIndexForCharacter(at: character)
    let rect = manager.lineFragmentRect(forGlyphAt: glyph, effectiveRange: nil)
    return text.convert(NSPoint(x: 0, y: rect.minY + text.textContainerOrigin.y), to: document).y
  }

  private static func views<T: NSView>(_ type: T.Type, in root: NSView) -> [T] {
    if let match = root as? T { return [match] }
    return root.subviews.flatMap { views(type, in: $0) }
  }
}
