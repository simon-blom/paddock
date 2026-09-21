import AppKit
import PaddockStudio
import SwiftUI

struct StudioSamplingRow: View {
  let dial: StudioState.Composer.Dial
  @Binding var value: Double
  let display: String

  var body: some View {
    VStack(spacing: 5) {
      HStack(alignment: .firstTextBaseline) {
        Text(dial.label)
        Spacer(minLength: 12)
        Text(display).foregroundStyle(.secondary).monospacedDigit().fixedSize()
      }
      StudioSamplingSlider(
        value: $value, bounds: dial.min...dial.max, step: dial.step, label: dial.label
      ).frame(height: 18)
    }
  }
}

/// SwiftUI's stepped Slider creates a tick for each value on macOS: 201 ticks
/// on Top-k. Keep an ordinary AppKit slider and quantize its actions instead.
/// AppKit owns the knob, tracking and focus. Only the bar uses our flat colors.
struct StudioSamplingSlider: NSViewRepresentable {
  @Binding var value: Double
  let bounds: ClosedRange<Double>
  let step: Double
  let label: String
  @Environment(\.isEnabled) private var enabled
  @Environment(\.colorScheme) private var colorScheme

  func makeNSView(context: Context) -> SamplingSliderControl {
    let slider = SamplingSliderControl()
    slider.setContentHuggingPriority(.defaultLow, for: .horizontal)
    return slider
  }
  func updateNSView(_ slider: SamplingSliderControl, context: Context) {
    slider.minValue = bounds.lowerBound
    slider.maxValue = bounds.upperBound
    slider.valueStep = step
    slider.altIncrementValue = step
    slider.doubleValue = min(bounds.upperBound, max(bounds.lowerBound, value))
    slider.isEnabled = enabled
    slider.trackFillColor = PaddockStyle.nsColor("accent", dark: colorScheme == .dark)
    (slider.cell as? SamplingSliderCell)?.trackColor = PaddockStyle.nsColor(
      "borderStrong", dark: colorScheme == .dark)
    slider.needsDisplay = true
    slider.setAccessibilityLabel(label)
    slider.onChange = { value = $0 }
  }
  func sizeThatFits(
    _ proposal: ProposedViewSize, nsView: SamplingSliderControl, context: Context
  ) -> CGSize? {
    CGSize(width: proposal.width ?? 288, height: 18)
  }
}

final class SamplingSliderControl: NSSlider {
  var valueStep = 1.0
  var onChange: ((Double) -> Void)?
  override init(frame: NSRect) {
    super.init(frame: frame)
    cell = SamplingSliderCell()
    sliderType = .linear
    isVertical = false
    controlSize = .small
    isContinuous = true
    numberOfTickMarks = 0
    allowsTickMarkValuesOnly = false
    target = self
    action = #selector(changed)
  }
  required init?(coder: NSCoder) { nil }

  override var acceptsFirstResponder: Bool { isEnabled }
  override func mouseDown(with event: NSEvent) {
    if isEnabled { window?.makeFirstResponder(self) }
    super.mouseDown(with: event)
  }

  @objc private func changed() { commit(doubleValue) }

  /// The shared sampler still owns ranges/steps. An untouched model default
  /// is never rewritten merely because the control was mounted or refreshed.
  func commit(_ proposed: Double) {
    guard isEnabled, proposed.isFinite, valueStep.isFinite, valueStep > 0 else { return }
    let snapped = minValue + ((proposed - minValue) / valueStep).rounded() * valueStep
    doubleValue = min(maxValue, max(minValue, snapped))
    onChange?(doubleValue)
    NSAccessibility.post(element: self, notification: .valueChanged)
  }

  override func keyDown(with event: NSEvent) {
    guard !event.modifierFlags.contains(.command), !event.modifierFlags.contains(.control) else {
      super.keyDown(with: event)
      return
    }
    switch event.keyCode {
    case 123, 125: commit(doubleValue - valueStep)  // left / down
    case 124, 126: commit(doubleValue + valueStep)  // right / up
    case 115: commit(minValue)  // home
    case 119: commit(maxValue)  // end
    default: super.keyDown(with: event)
    }
  }
  override func accessibilityPerformIncrement() -> Bool {
    guard isEnabled else { return false }
    commit(doubleValue + valueStep)
    return true
  }
  override func accessibilityPerformDecrement() -> Bool {
    guard isEnabled else { return false }
    commit(doubleValue - valueStep)
    return true
  }
  override func setAccessibilityValue(_ value: Any?) {
    if let value = value as? NSNumber { commit(value.doubleValue) }
  }
}

/// The system's unfilled track can disappear on an opaque dark popover (its
/// default treatment expects a material). Paint a complete, quiet rail; retain
/// the native thumb, hit geometry and keyboard focus ring unchanged.
final class SamplingSliderCell: NSSliderCell {
  var trackColor = NSColor.separatorColor
  override func drawBar(inside rect: NSRect, flipped: Bool) {
    let rail = NSRect(x: rect.minX, y: rect.midY - 1.5, width: rect.width, height: 3)
    trackColor.withAlphaComponent(isEnabled ? 1 : 0.4).setFill()
    NSBezierPath(roundedRect: rail, xRadius: 1.5, yRadius: 1.5).fill()
    let thumb = knobRect(flipped: flipped)
    let filled = NSRect(
      x: rail.minX, y: rail.minY,
      width: min(rail.width, max(0, thumb.midX - rail.minX)), height: rail.height)
    let color = (controlView as? NSSlider)?.trackFillColor ?? .labelColor
    color.withAlphaComponent(isEnabled ? 1 : 0.4).setFill()
    NSBezierPath(roundedRect: filled, xRadius: 1.5, yRadius: 1.5).fill()
  }
}
