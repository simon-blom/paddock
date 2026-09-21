import AppKit

/// The native equivalent of web Studio's dictation widget: layout, not content.
/// This separate TextKit 2 graph has no selection, responder, undo or editable
/// storage. Its transparent prefix gives the suffix exactly the draft's wrapping.
/// Only a changed suffix is replaced as recognition streams in; the graph is
/// released when the utterance ends. No WebKit and no IME marked-text workaround.
@MainActor final class DraftDictationOverlay: NSView {
  let content = NSTextContentStorage()
  let layout = NSTextLayoutManager()
  let container = NSTextContainer(size: .zero)
  private(set) var prefix = ""
  private(set) var suffix = ""
  private var prefixLength = 0
  private var textFont: NSFont?
  private var paragraph: NSParagraphStyle?
  private var inset = NSSize.zero
  private(set) var contentHeight: CGFloat = 0

  override init(frame: NSRect) {
    super.init(frame: frame)
    content.textStorage = NSTextStorage()
    content.addTextLayoutManager(layout)
    layout.textContainer = container
    setAccessibilityElement(false)
    setAccessibilityChildren([])
  }
  required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
  override var isFlipped: Bool { true }
  override func hitTest(_ point: NSPoint) -> NSView? { nil }

  /// Web trims leading whitespace, keeps a provisional trailing space, and
  /// recomputes the boundary against the current draft on every edit.
  static func suffix(after draft: String, provisional: String) -> String {
    let said = provisional.drop(while: \.isWhitespace)
    guard !said.isEmpty else { return "" }
    return (draft.last.map { $0.isWhitespace ? "" : " " } ?? "") + said
  }

  func update(editor: DraftTextView, provisional: String) {
    let draft = editor.string
    let next = Self.suffix(after: draft, provisional: provisional)
    let font = editor.font ?? .systemFont(ofSize: StudioDraftEditor.textFontSize)
    let style = editor.defaultParagraphStyle ?? NSParagraphStyle.default
    let width = editor.textContainer?.size.width ?? editor.bounds.width
    let padding = editor.textContainer?.lineFragmentPadding ?? 5
    guard let storage = content.textStorage else { return }
    let restyle = textFont != font || paragraph != style
    let relayout =
      container.size.width != width || container.lineFragmentPadding != padding
      || inset != editor.textContainerInset
    guard draft != prefix || next != suffix || restyle || relayout else { return }
    inset = editor.textContainerInset
    container.size = NSSize(width: width, height: .greatestFiniteMagnitude)
    container.lineFragmentPadding = padding
    if draft != prefix || restyle {
      prefix = draft
      prefixLength = (draft as NSString).length
      storage.setAttributedString(
        NSAttributedString(
          string: draft,
          attributes: [
            .font: font, .paragraphStyle: style, .foregroundColor: NSColor.clear,
          ]))
    }
    storage.replaceCharacters(
      in: NSRange(location: prefixLength, length: storage.length - prefixLength),
      with: NSAttributedString(
        string: next,
        attributes: [
          .font: font, .paragraphStyle: style, .foregroundColor: NSColor.secondaryLabelColor,
        ]))
    suffix = next
    textFont = font
    paragraph = style
    // TextKit keeps unchanged paragraphs laid out; streaming only invalidates
    // the tail. The height belongs to decoration, never to the draft document.
    layout.ensureLayout(for: content.documentRange)
    contentHeight = ceil(layout.usageBoundsForTextContainer.height + inset.height * 2)
    frame = NSRect(x: 0, y: 0, width: editor.bounds.width, height: contentHeight)
    needsDisplay = true
  }

  override func draw(_ dirtyRect: NSRect) {
    guard let context = NSGraphicsContext.current?.cgContext else { return }
    let start = content.location(content.documentRange.location, offsetBy: prefixLength)
    layout.enumerateTextLayoutFragments(from: start, options: []) {
      fragment in
      let origin = NSPoint(
        x: fragment.layoutFragmentFrame.minX + self.inset.width,
        y: fragment.layoutFragmentFrame.minY + self.inset.height)
      let bounds = fragment.renderingSurfaceBounds.offsetBy(dx: origin.x, dy: origin.y)
      if bounds.intersects(dirtyRect) { fragment.draw(at: origin, in: context) }
      return fragment.layoutFragmentFrame.minY <= dirtyRect.maxY
    }
  }
  override func viewDidChangeEffectiveAppearance() {
    super.viewDidChangeEffectiveAppearance()
    needsDisplay = true
  }
}
