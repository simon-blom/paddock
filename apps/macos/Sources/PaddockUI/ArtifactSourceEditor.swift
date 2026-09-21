import AppKit
import SwiftUI

/// One session per artifact pane, retained while Source is hidden by Preview.
/// Only executable HTML preview uses WebKit; source editing never does.
struct ArtifactSourceEditor: NSViewRepresentable {
  let session: ArtifactSourceSession
  let identity: String
  @Binding var text: String
  let language: String
  let readOnly: Bool
  let save: () -> Void
  @Environment(\.colorScheme) private var colorScheme

  func makeCoordinator() -> ArtifactSourceSession { session }
  func makeNSView(context: Context) -> NSScrollView { session.scrollView }
  func updateNSView(_ view: NSScrollView, context: Context) {
    session.onChange = { text = $0 }
    session.textView.save = save
    session.update(
      identity: identity, text: text, language: language, readOnly: readOnly,
      dark: colorScheme == .dark)
  }
  static func dismantleNSView(_ view: NSScrollView, coordinator: ArtifactSourceSession) {
    coordinator.suspend()
  }
}

@MainActor final class ArtifactSourceSession: NSObject, ObservableObject, NSTextViewDelegate {
  let scrollView = NSScrollView()
  let textView = ArtifactCodeTextView(usingTextLayoutManager: true)
  private(set) var syntax: ArtifactSourceSyntax?
  private(set) var revision = 0
  private(set) var appliedRevision = -1
  private var identity = ""
  private var language = ""
  private var dark: Bool?
  private var worker: Task<ArtifactSourceSyntax, Error>?
  private var delivery: Task<Void, Never>?
  private var gutter: ArtifactSourceGutter!
  private var synchronizing = false
  private var repaintQueued = false
  private var firstUnparsedEdit = 0
  var onChange: ((String) -> Void)?

  override init() {
    super.init()
    scrollView.hasVerticalScroller = true
    scrollView.hasHorizontalScroller = false
    scrollView.autohidesScrollers = true
    scrollView.borderType = .noBorder
    PaddockScrollbars.install(on: scrollView)
    scrollView.drawsBackground = true
    scrollView.clipsToBounds = true
    textView.frame = NSRect(x: 0, y: 0, width: 400, height: 300)
    textView.minSize = .zero
    textView.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
    textView.isVerticallyResizable = true
    textView.isHorizontallyResizable = false
    textView.autoresizingMask = [.width]
    textView.textContainer?.widthTracksTextView = true
    textView.textContainer?.containerSize = NSSize(
      width: 400, height: CGFloat.greatestFiniteMagnitude)
    textView.textContainerInset = NSSize(width: 6, height: 8)
    textView.textContainer?.lineFragmentPadding = 2
    textView.isRichText = false
    textView.importsGraphics = false
    textView.allowsUndo = true
    textView.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
    let paragraph = NSMutableParagraphStyle()
    paragraph.minimumLineHeight = 18
    paragraph.maximumLineHeight = 18
    paragraph.defaultTabInterval =
      ("  " as NSString).size(withAttributes: [.font: textView.font!]).width
    paragraph.tabStops = []
    textView.defaultParagraphStyle = paragraph
    textView.typingAttributes = [.font: textView.font!, .paragraphStyle: paragraph]
    textView.isAutomaticQuoteSubstitutionEnabled = false
    textView.isAutomaticDashSubstitutionEnabled = false
    textView.isAutomaticTextReplacementEnabled = false
    textView.isAutomaticSpellingCorrectionEnabled = false
    textView.isContinuousSpellCheckingEnabled = false
    textView.isGrammarCheckingEnabled = false
    textView.isAutomaticLinkDetectionEnabled = false
    textView.isAutomaticDataDetectionEnabled = false
    textView.isAutomaticTextCompletionEnabled = false
    textView.usesFindBar = true
    textView.isIncrementalSearchingEnabled = true
    textView.setAccessibilityLabel("Artifact source editor")
    textView.setAccessibilityIdentifier("artifact-source-editor")
    textView.delegate = self
    textView.textStorage?.delegate = textView
    textView.onStorageEdit = { [weak self] location in
      guard let self else { return }
      self.firstUnparsedEdit = min(self.firstUnparsedEdit, location)
    }
    textView.contextAt = { [weak self] offset in
      guard let self else { return nil }
      return ArtifactSourceSyntax.context(
        self.textView.string, offset: offset, language: self.language,
        current: self.syntax?.language == ArtifactSourceSyntax.normalize(self.language)
          ? self.syntax : nil,
        unchangedThrough: self.firstUnparsedEdit)
    }
    textView.matchAt = { [weak self] offset in self?.matchingBracket(at: offset) }
    scrollView.documentView = textView
    gutter = ArtifactSourceGutter(scrollView: scrollView, orientation: .verticalRuler)
    gutter.clipsToBounds = true
    gutter.ruleThickness = 40
    gutter.session = self
    scrollView.verticalRulerView = gutter
    scrollView.hasVerticalRuler = true
    scrollView.rulersVisible = true
    scrollView.contentView.postsBoundsChangedNotifications = true
    NotificationCenter.default.addObserver(
      self, selector: #selector(scrolled), name: NSView.boundsDidChangeNotification,
      object: scrollView.contentView)
    // Rendering-only attributes don't touch the undo stack or invalidate glyph
    // geometry. Never access NSTextView.layoutManager: that opts out of TK2.
    textView.textLayoutManager?.renderingAttributesValidator = { [weak self] manager, fragment in
      self?.highlight(manager, fragment)
    }
  }

  deinit {
    worker?.cancel()
    delivery?.cancel()
    NotificationCenter.default.removeObserver(self)
  }

  func update(identity: String, text: String, language: String, readOnly: Bool, dark: Bool) {
    let newDocument = self.identity != identity
    let newLanguage = self.language != language
    self.identity = identity
    self.language = language
    textView.isEditable = !readOnly
    textView.setAccessibilityHelp(readOnly ? "Older version. Select Latest to edit." : "")
    let changed = textView.string != text
    if newDocument || changed {
      synchronizing = true
      let selection = textView.selectedRanges
      let origin = scrollView.contentView.bounds.origin
      textView.resetEditingState()
      firstUnparsedEdit = 0
      textView.string = text
      // External replacements (Revert, version change) are not user keystrokes.
      // Local binding echoes and save acknowledgements never enter this path.
      textView.undoManager?.removeAllActions()
      let length = (text as NSString).length
      textView.selectedRanges =
        newDocument
        ? [NSValue(range: NSRange(location: 0, length: 0))]
        : selection.map {
          let range = $0.rangeValue
          let start = min(range.location, length)
          return NSValue(range: NSRange(location: start, length: min(range.length, length - start)))
        }
      scrollView.contentView.scroll(to: newDocument ? .zero : origin)
      synchronizing = false
      revision += 1
    }
    if self.dark != dark {
      self.dark = dark
      let background = PaddockStyle.nsColor("canvas", dark: dark)
      scrollView.backgroundColor = background
      textView.backgroundColor = background
      textView.textColor = PaddockStyle.nsColor("primary", dark: dark)
      textView.insertionPointColor = PaddockStyle.nsColor("primary", dark: dark)
      refreshRendering()
    }
    if newDocument || changed || newLanguage || appliedRevision != revision { schedule() }
  }

  func textDidChange(_ notification: Notification) {
    guard !synchronizing else { return }
    revision += 1
    onChange?(textView.string)
    schedule()
  }

  func textViewDidChangeSelection(_ notification: Notification) {
    guard !synchronizing else { return }
    paintViewport()
    textView.needsDisplay = true
  }

  private func matchingBracket(at offset: Int) -> (Int, Int)? {
    guard appliedRevision == revision, let syntax, syntax.length == textView.textStorage?.length
    else { return nil }
    if let partner = syntax.brackets[offset] { return (offset, partner) }
    if offset > 0, let partner = syntax.brackets[offset - 1] { return (offset - 1, partner) }
    return nil
  }

  func suspend() {
    worker?.cancel()
    delivery?.cancel()
    worker = nil
    delivery = nil
    onChange = nil
    textView.save = nil
  }

  private func schedule() {
    worker?.cancel()
    delivery?.cancel()
    let text = textView.string
    let language = language
    let previous = syntax
    let expected = revision
    // Snapshot immutable data on main, tokenize off main. Every pane has its own
    // cancellable worker, so a large artifact cannot queue other Compare lanes.
    let task = Task.detached(priority: .userInitiated) {
      try await Task.sleep(for: .milliseconds(35))
      return try ArtifactSourceSyntax.parse(text, language: language, previous: previous)
    }
    worker = task
    delivery = Task { [weak self] in
      guard let parsed = try? await task.value, !Task.isCancelled, let self,
        self.revision == expected
      else { return }
      self.syntax = parsed
      self.appliedRevision = expected
      self.firstUnparsedEdit = .max
      self.textView.completionWords = parsed.words
      self.worker = nil
      self.delivery = nil
      let digits = max(3, String(parsed.lines.count).count)
      self.gutter.ruleThickness = CGFloat(digits * 8 + 16)
      self.refreshRendering()
    }
    gutter.needsDisplay = true
  }

  private func refreshRendering() {
    if let manager = textView.textLayoutManager,
      let document = manager.textContentManager?.documentRange
    {
      manager.invalidateRenderingAttributes(for: document)
    }
    paintViewport()
    textView.needsDisplay = true
    gutter.needsDisplay = true
  }
  private func paintViewport() {
    guard let manager = textView.textLayoutManager,
      let viewport = manager.textViewportLayoutController.viewportRange
    else { return }
    // Already-laid-out fragments are not guaranteed to call the validator after
    // asynchronous token delivery. Explicitly repaint this small visible set.
    manager.enumerateTextLayoutFragments(from: viewport.location, options: []) { fragment in
      guard fragment.rangeInElement.location.compare(viewport.endLocation) == .orderedAscending
      else { return false }
      self.highlight(manager, fragment)
      return true
    }
  }
  @objc private func scrolled() {
    gutter.needsDisplay = true
    guard !repaintQueued else { return }
    repaintQueued = true
    // Bounds notifications can occur inside viewport layout. Coalesce until
    // that pass finishes rather than re-entering TextKit from its notification.
    DispatchQueue.main.async { [weak self] in
      guard let self else { return }
      self.repaintQueued = false
      self.paintViewport()
      self.textView.needsDisplay = true
    }
  }

  private func highlight(_ manager: NSTextLayoutManager, _ fragment: NSTextLayoutFragment) {
    guard let content = manager.textContentManager else { return }
    let range = fragment.rangeInElement
    manager.setRenderingAttributes(
      [.foregroundColor: PaddockStyle.nsColor("primary", dark: dark ?? false)], for: range)
    guard let syntax, appliedRevision == revision else { return }
    let start = content.offset(from: content.documentRange.location, to: range.location)
    let end = content.offset(from: content.documentRange.location, to: range.endLocation)
    guard start >= 0, end <= syntax.length else { return }
    for line in syntax.line(at: start)...syntax.line(at: max(start, end - 1)) {
      for token in syntax.lines[line].tokens {
        let lo = max(start, syntax.starts[line] + token.range.location)
        let hi = min(end, syntax.starts[line] + NSMaxRange(token.range))
        guard hi > lo,
          let a = content.location(content.documentRange.location, offsetBy: lo),
          let b = content.location(a, offsetBy: hi - lo),
          let tokenRange = NSTextRange(location: a, end: b)
        else { continue }
        manager.setRenderingAttributes([.foregroundColor: color(token.kind)], for: tokenRange)
      }
    }
    if textView.selectedRange().length == 0,
      let pair = matchingBracket(at: textView.selectedRange().location)
    {
      for offset in [pair.0, pair.1] where offset >= start && offset < end {
        if let a = content.location(content.documentRange.location, offsetBy: offset),
          let b = content.location(a, offsetBy: 1), let bracket = NSTextRange(location: a, end: b)
        {
          manager.addRenderingAttribute(
            .underlineStyle, value: NSUnderlineStyle.single.rawValue, for: bracket)
          manager.addRenderingAttribute(
            .backgroundColor, value: PaddockStyle.nsColor("elevated", dark: dark ?? false),
            for: bracket)
        }
      }
    }
  }

  private func color(_ kind: ArtifactSourceSyntax.Kind) -> NSColor {
    // Syntax is content, not chrome. Explicit light/dark palettes avoid tinted
    // wallpaper materials and retain contrast in either appearance.
    let rgb: UInt32
    switch kind {
    case .comment: rgb = dark == true ? 0x92969F : 0x626973
    case .string: rgb = dark == true ? 0xA8D19D : 0x326827
    case .number: rgb = dark == true ? 0xDEBA85 : 0x895413
    case .keyword: rgb = dark == true ? 0xCFACED : 0x7942A1
    case .tag: rgb = dark == true ? 0x93BFEF : 0x245EA0
    case .attribute: rgb = dark == true ? 0xDACB9C : 0x795B20
    case .bracket: return PaddockStyle.nsColor("primary", dark: dark ?? false)
    }
    return NSColor(
      srgbRed: CGFloat(rgb >> 16 & 255) / 255, green: CGFloat(rgb >> 8 & 255) / 255,
      blue: CGFloat(rgb & 255) / 255, alpha: 1)
  }
}

/// Enumerate laid-out viewport fragments only. No whole-document glyph layout,
/// no line number on wrapped continuations, no independent scrolling surface.
@MainActor private final class ArtifactSourceGutter: NSRulerView {
  weak var session: ArtifactSourceSession?
  override var isFlipped: Bool { true }
  override func drawHashMarksAndLabels(in rect: NSRect) {
    guard let session else { return }
    // AppKit supplies a document invalidation rect, not necessarily the ruler's
    // bounds. macOS no longer clips every NSView by default: never let this
    // opaque fill cover the editor or the SwiftUI header above it.
    NSGraphicsContext.saveGraphicsState()
    defer { NSGraphicsContext.restoreGraphicsState() }
    NSBezierPath(rect: bounds).addClip()
    session.textView.backgroundColor.setFill()
    bounds.intersection(rect).fill()
    guard let syntax = session.syntax, session.appliedRevision == session.revision,
      let manager = session.textView.textLayoutManager,
      let content = manager.textContentManager,
      let viewport = manager.textViewportLayoutController.viewportRange
    else { return }
    let attrs: [NSAttributedString.Key: Any] = [
      .font: NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular),
      .foregroundColor: NSColor.secondaryLabelColor,
    ]
    manager.enumerateTextLayoutFragments(from: viewport.location, options: []) { fragment in
      let offset = content.offset(
        from: content.documentRange.location, to: fragment.rangeInElement.location)
      let index = syntax.line(at: offset)
      guard syntax.starts[index] == offset else { return true }
      let origin = self.convert(
        NSPoint(
          x: 0, y: fragment.layoutFragmentFrame.minY + session.textView.textContainerOrigin.y),
        from: session.textView)
      if origin.y > self.bounds.maxY { return false }
      if origin.y + fragment.layoutFragmentFrame.height >= self.bounds.minY {
        let value = "\(index + 1)" as NSString
        value.draw(
          at: NSPoint(
            x: self.ruleThickness - value.size(withAttributes: attrs).width - 8, y: origin.y + 3),
          withAttributes: attrs)
      }
      return true
    }
  }
}
