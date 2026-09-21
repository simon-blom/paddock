import SwiftUI

/// Intrinsic-width choices, in reading order. Unlike an adaptive grid, the
/// last row never gets stretched cells or holes between short labels.
struct SettingsChoiceLayout: Layout {
  var spacing: CGFloat = 8

  static func positions(sizes: [CGSize], width: CGFloat, spacing: CGFloat) -> [CGRect] {
    var x: CGFloat = 0
    var y: CGFloat = 0
    var rowHeight: CGFloat = 0
    return sizes.map { size in
      if x > 0 && x + size.width > width {
        x = 0
        y += rowHeight + spacing
        rowHeight = 0
      }
      let rect = CGRect(origin: CGPoint(x: x, y: y), size: size)
      x += size.width + spacing
      rowHeight = max(rowHeight, size.height)
      return rect
    }
  }
  private func rects(_ subviews: Subviews, width: CGFloat?) -> [CGRect] {
    Self.positions(
      sizes: subviews.map { $0.sizeThatFits(.unspecified) },
      width: width ?? .infinity, spacing: spacing)
  }
  func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
    let rects = rects(subviews, width: proposal.width)
    return CGSize(width: rects.map(\.maxX).max() ?? 0, height: rects.map(\.maxY).max() ?? 0)
  }
  func placeSubviews(
    in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()
  ) {
    for (view, rect) in zip(subviews, rects(subviews, width: bounds.width)) {
      view.place(
        at: CGPoint(x: bounds.minX + rect.minX, y: bounds.minY + rect.minY),
        anchor: .topLeading, proposal: ProposedViewSize(rect.size))
    }
  }
}
