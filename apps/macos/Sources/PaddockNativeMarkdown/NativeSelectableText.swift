import AppKit
import SwiftUI

private struct ConversationTextIDKey: EnvironmentKey { static let defaultValue = "" }
extension EnvironmentValues {
  public var conversationTextID: String {
    get { self[ConversationTextIDKey.self] }
    set { self[ConversationTextIDKey.self] = newValue }
  }
}

/// Actual TextKit text, not independently selectable SwiftUI Text fragments.
/// Used for literal user messages and code so the transcript selection owner
/// can span those surfaces without flattening their visual presentation.
public struct NativeSelectableText: NSViewRepresentable {
  let text: String
  var attributed: AttributedString?
  var size: CGFloat
  var monospaced: Bool
  var wraps: Bool
  public init(
    _ text: String, attributed: AttributedString? = nil, size: CGFloat = 15,
    monospaced: Bool = false, wraps: Bool = true
  ) {
    self.text = text
    self.attributed = attributed
    self.size = size
    self.monospaced = monospaced
    self.wraps = wraps
  }
  public func makeNSView(context: Context) -> NSTextView {
    let view = NSTextView(frame: .zero)
    view.isEditable = false
    view.isSelectable = true
    view.drawsBackground = false
    view.textContainerInset = .zero
    view.textContainer?.lineFragmentPadding = 0
    view.isVerticallyResizable = false
    view.isHorizontallyResizable = false
    return view
  }
  public func updateNSView(_ view: NSTextView, context: Context) {
    view.identifier = .init(context.environment.conversationTextID)
    let value = attributed.map(NSAttributedString.init) ?? NSAttributedString(string: text)
    let styled = NSMutableAttributedString(attributedString: value)
    let full = NSRange(location: 0, length: styled.length)
    styled.addAttribute(
      .font,
      value: monospaced
        ? NSFont.monospacedSystemFont(ofSize: size, weight: .regular)
        : NSFont.systemFont(ofSize: size), range: full)
    if attributed == nil {
      styled.addAttribute(.foregroundColor, value: NSColor.labelColor, range: full)
    }
    if view.textStorage?.isEqual(to: styled) != true {
      let range = view.selectedRange()
      let previous = view.string as NSString
      view.textStorage?.setAttributedString(styled)
      if NSMaxRange(range) <= previous.length, NSMaxRange(range) <= styled.length,
        previous.substring(with: range) == (styled.string as NSString).substring(with: range)
      {
        view.setSelectedRange(range)
      }
    }
    view.textContainer?.widthTracksTextView = wraps
  }
  public func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSTextView, context: Context)
    -> CGSize?
  {
    guard let container = nsView.textContainer, let manager = nsView.layoutManager else {
      return nil
    }
    let width = wraps ? max(1, proposal.width ?? 760) : 100_000
    container.containerSize = CGSize(width: width, height: .greatestFiniteMagnitude)
    manager.ensureLayout(for: container)
    let size = manager.usedRect(for: container).size
    return CGSize(
      width: wraps ? width : max(1, ceil(size.width)),
      height: max(size.height, size.height == 0 ? self.size + 4 : 0))
  }
}
