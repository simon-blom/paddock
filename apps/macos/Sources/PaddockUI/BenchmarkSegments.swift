import AppKit
import SwiftUI

/// AppKit owns selection, keyboard navigation and VoiceOver. Unlike a SwiftUI
/// segmented Picker's intrinsic-width content, these segments fill the shared
/// form column, without adding a second visible field label.
struct BenchmarkSegments<Value: Equatable>: NSViewRepresentable {
  let title: String
  let options: [(Value, String)]
  @Binding var selection: Value
  @Environment(\.isEnabled) private var isEnabled

  func makeCoordinator() -> Coordinator { Coordinator(self) }
  func makeNSView(context: Context) -> NSSegmentedControl {
    let control = NSSegmentedControl(
      labels: options.map(\.1), trackingMode: .selectOne,
      target: context.coordinator, action: #selector(Coordinator.select(_:)))
    control.segmentStyle = .rounded
    control.segmentDistribution = .fillEqually
    control.setContentHuggingPriority(.defaultLow, for: .horizontal)
    updateNSView(control, context: context)
    return control
  }
  func updateNSView(_ control: NSSegmentedControl, context: Context) {
    context.coordinator.parent = self
    control.selectedSegment = options.firstIndex { $0.0 == selection } ?? -1
    control.isEnabled = isEnabled
    control.setAccessibilityLabel(title)
  }
  func sizeThatFits(_ proposal: ProposedViewSize, nsView: NSSegmentedControl, context: Context)
    -> CGSize?
  {
    CGSize(width: proposal.width ?? nsView.intrinsicContentSize.width, height: 30)
  }
  @MainActor final class Coordinator: NSObject {
    var parent: BenchmarkSegments
    init(_ parent: BenchmarkSegments) { self.parent = parent }
    @objc func select(_ sender: NSSegmentedControl) {
      guard sender.isEnabled, parent.options.indices.contains(sender.selectedSegment) else {
        return
      }
      parent.selection = parent.options[sender.selectedSegment].0
    }
  }
}
