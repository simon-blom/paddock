import PaddockDesign
import SwiftUI

typealias PaddockScrollView<Content: View> = PaddockDesign.PaddockScrollView<Content>
typealias PaddockTextEditor = PaddockDesign.PaddockTextEditor
typealias PaddockScrollbars = PaddockDesign.PaddockScrollbars
typealias PaddockScroller = PaddockDesign.PaddockScroller
typealias PaddockScrollStyle = PaddockDesign.PaddockScrollStyle

/// A bounded owner for controls such as Form that create their own scroller.
/// Styling stays inside this host and cannot reach other windows or panels.
struct PaddockScrollRegion<Content: View>: NSViewRepresentable {
  @ViewBuilder var content: Content
  func makeNSView(context: Context) -> PaddockScrollHostingView {
    PaddockScrollHostingView(rootView: AnyView(content.environment(\.self, context.environment)))
  }
  func updateNSView(_ view: PaddockScrollHostingView, context: Context) {
    view.rootView = AnyView(content.environment(\.self, context.environment))
  }
  func sizeThatFits(
    _ proposal: ProposedViewSize, nsView: PaddockScrollHostingView, context: Context
  ) -> CGSize? {
    guard let width = proposal.width, let height = proposal.height else { return nil }
    return CGSize(width: width, height: height)
  }
}
