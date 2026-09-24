import AppKit
import PaddockDesign
import SwiftUI

/// Own the hosting boundary so a library text-storage replacement cannot move
/// an existing selection to the end of a streaming response. No global view
/// introspection, delegate replacement or modifications to dependency sources.
struct SelectableMarkdownSurface<Content: View>: NSViewRepresentable {
  let content: Content
  var unsureWords: [String]

  init(unsureWords: [String] = [], @ViewBuilder content: () -> Content) {
    self.content = content()
    self.unsureWords = unsureWords
  }

  func makeNSView(context: Context) -> SelectionHostingView {
    SelectionHostingView(rootView: AnyView(content.environment(\.self, context.environment)))
  }

  func updateNSView(_ view: SelectionHostingView, context: Context) {
    #if DEBUG
      view.contentUpdates += 1
    #endif
    view.identifier = .init(context.environment.conversationTextID)
    view.confidence.words = unsureWords
    view.preserveSelectionForUpdate()
    view.rootView = AnyView(content.environment(\.self, context.environment))
    view.needsLayout = true
  }
}

final class SelectionHostingView: PaddockScrollHostingView {
  #if DEBUG
    var contentUpdates = 0
  #endif
  let confidence = NativeConfidenceDecoration()
  private struct Selection {
    weak var view: NSTextView?
    let text: String
    let ranges: [NSValue]
  }
  private var pending: Selection?

  func preserveSelectionForUpdate() {
    guard pending == nil, let text = textSurface(in: self),
      text.selectedRanges.contains(where: { $0.rangeValue.length > 0 })
    else { return }
    pending = Selection(view: text, text: text.string, ranges: text.selectedRanges)
  }

  override func layout() {
    super.layout()
    confidence.reconcile(self)
    guard let selection = pending else { return }
    pending = nil
    guard let view = selection.view else { return }
    // Restore only unchanged selected text. If an edit invalidates the selected
    // region, leave AppKit's selection alone instead of selecting different text.
    let next = view.string as NSString
    let previous = selection.text as NSString
    guard
      selection.ranges.allSatisfy({ value in
        let range = value.rangeValue
        return NSMaxRange(range) <= next.length
          && next.substring(with: range) == previous.substring(with: range)
      })
    else { return }
    view.selectedRanges = selection.ranges
  }

  private func textSurface(in view: NSView) -> NSTextView? {
    if let text = view as? NSTextView { return text }
    return view.subviews.lazy.compactMap { self.textSurface(in: $0) }.first
  }
}
