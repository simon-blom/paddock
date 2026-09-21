import SwiftUI

/// Full-height catalog split with chrome clearance inside the columns, never
/// above the divider. Keep the user's preferred width while narrowing.
struct CatalogColumns<Master: View, Detail: View>: View {
  var listLimits: ClosedRange<CGFloat> = 240...400
  @ViewBuilder var master: Master
  @ViewBuilder var detail: Detail
  @State private var preferredWidth: CGFloat = 310
  @Environment(\.workspaceCatalogContentInset) private var contentInset

  static var detailMinimum: CGFloat { 350 }

  static func listWidth(
    preferred: CGFloat, available: CGFloat, limits: ClosedRange<CGFloat>
  ) -> CGFloat {
    let maximum = max(
      limits.lowerBound,
      min(limits.upperBound, available - WorkspacePanelMetrics.gap - detailMinimum))
    return min(max(preferred, limits.lowerBound), maximum)
  }

  var body: some View {
    GeometryReader { geometry in
      let width = Self.listWidth(
        preferred: preferredWidth, available: geometry.size.width, limits: listLimits)
      let maximum = Self.listWidth(
        preferred: listLimits.upperBound, available: geometry.size.width, limits: listLimits)
      HStack(spacing: 0) {
        master.padding(.top, contentInset).frame(width: width, height: geometry.size.height)
        WorkspacePanelGrip(
          width: Binding(get: { width }, set: { preferredWidth = $0 }),
          range: listLimits.lowerBound...maximum, label: "Resize model list"
        ).accessibilityIdentifier("catalog-column-grip")
        detail.padding(.top, contentInset).frame(
          minWidth: Self.detailMinimum, maxWidth: .infinity, maxHeight: .infinity)
      }
    }.frame(minWidth: listLimits.lowerBound + WorkspacePanelMetrics.gap + Self.detailMinimum)
  }
}
