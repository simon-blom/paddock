import AppKit
import PaddockStudio
import SwiftUI

/// An AppKit slider with waveform-specific pointer mapping and accessibility.
/// The cell caches its bitmaps; playback changes clipping, not envelope layout.
struct NativeWaveformTrack: NSViewRepresentable {
  let waveform: AudioWaveform?
  let duration: Double
  let position: Double
  var notices: [StudioState.Speech.Guard] = []
  var enabled = true
  var onSeek: (Double) -> Void
  var onToggle: () -> Void
  func makeNSView(context: Context) -> WaveformSlider {
    let slider = WaveformSlider(frame: .zero)
    slider.cell = WaveformSliderCell()
    slider.isContinuous = true
    slider.target = slider
    slider.action = #selector(WaveformSlider.changed)
    slider.focusRingType = .exterior
    slider.setAccessibilityLabel("Recording position")
    slider.setAccessibilityIdentifier("audio-waveform")
    return slider
  }
  func updateNSView(_ slider: WaveformSlider, context: Context) {
    slider.onSeek = onSeek
    slider.onToggle = onToggle
    slider.minValue = 0
    slider.maxValue = duration.isFinite && duration > 0 ? duration : 1
    slider.isEnabled = enabled && duration.isFinite && duration > 0
    if !slider.tracking {
      slider.doubleValue = position.isFinite ? min(slider.maxValue, max(0, position)) : 0
    }
    if let cell = slider.cell as? WaveformSliderCell {
      cell.waveform = waveform
      cell.notices = notices
    }
    slider.needsDisplay = true
  }
}

@MainActor final class WaveformSlider: NSSlider {
  override var intrinsicContentSize: NSSize { NSSize(width: NSView.noIntrinsicMetric, height: 48) }
  var onSeek: ((Double) -> Void)?
  var onToggle: (() -> Void)?
  private(set) var tracking = false
  private var hover: Double?
  private var mouseArea: NSTrackingArea?
  override func isAccessibilityElement() -> Bool { true }
  // Expose one slider, not both this waveform and NSSlider's stock cell proxy.
  override func accessibilityChildren() -> [Any]? { [] }
  override func accessibilityRole() -> NSAccessibility.Role? { .slider }
  override func accessibilityValue() -> Any? { doubleValue }
  override func accessibilityMinValue() -> Any? { minValue }
  override func accessibilityMaxValue() -> Any? { maxValue }
  override func accessibilityOrientation() -> NSAccessibilityOrientation { .horizontal }
  override func accessibilityValueDescription() -> String? {
    isEnabled ? "\(speechClock(doubleValue)) of \(speechClock(maxValue))" : "Duration unavailable"
  }
  override func accessibilityHelp() -> String? {
    let instructions = "Drag to seek. Arrow keys move five seconds. Space plays or pauses."
    guard let note = (cell as? WaveformSliderCell)?.noticeDescription(at: doubleValue) else {
      return instructions
    }
    return "\(instructions) \(note)"
  }
  override func setAccessibilityValue(_ value: Any?) {
    guard isEnabled, let value = value as? NSNumber, value.doubleValue.isFinite else { return }
    doubleValue = min(maxValue, max(minValue, value.doubleValue))
    changed()
  }
  override func accessibilityPerformIncrement() -> Bool { adjust(5) }
  override func accessibilityPerformDecrement() -> Bool { adjust(-5) }
  private func adjust(_ delta: Double) -> Bool {
    guard isEnabled else { return false }
    doubleValue = min(maxValue, max(minValue, doubleValue + delta))
    changed()
    return true
  }
  @objc func changed() { onSeek?(doubleValue) }
  override func mouseDown(with event: NSEvent) {
    guard isEnabled else { return }
    window?.makeFirstResponder(self)
    tracking = true
    seekPointer(event)
  }
  override func mouseDragged(with event: NSEvent) {
    guard tracking, isEnabled else { return }
    seekPointer(event)
  }
  override func mouseUp(with event: NSEvent) {
    guard tracking else { return }
    if isEnabled { seekPointer(event) }
    tracking = false
  }
  override func viewDidMoveToWindow() {
    super.viewDidMoveToWindow()
    if window == nil {
      tracking = false
      hover = nil
    }
  }
  private func seekPointer(_ event: NSEvent) {
    guard let cell = cell as? WaveformSliderCell else { return }
    let point = convert(event.locationInWindow, from: nil)
    let rect = cell.waveformRect
    doubleValue = min(1, max(0, (point.x - rect.minX) / max(1, rect.width))) * maxValue
    changed()
    needsDisplay = true
  }
  override func keyDown(with event: NSEvent) {
    guard isEnabled, event.modifierFlags.intersection([.command, .control, .option]).isEmpty else {
      super.keyDown(with: event)
      return
    }
    switch event.keyCode {
    case 49: onToggle?()
    case 123, 124:
      _ = adjust(event.keyCode == 123 ? -5 : 5)
    case 115:
      doubleValue = 0
      changed()
    case 119:
      doubleValue = maxValue
      changed()
    default: super.keyDown(with: event)
    }
  }
  override func updateTrackingAreas() {
    super.updateTrackingAreas()
    if let mouseArea { removeTrackingArea(mouseArea) }
    let area = NSTrackingArea(
      rect: .zero,
      options: [.inVisibleRect, .mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow],
      owner: self)
    addTrackingArea(area)
    mouseArea = area
  }
  override func mouseMoved(with event: NSEvent) {
    guard isEnabled, let cell = cell as? WaveformSliderCell else { return }
    let point = convert(event.locationInWindow, from: nil)
    let rect = cell.waveformRect
    let time = min(1, max(0, (point.x - rect.minX) / max(1, rect.width))) * maxValue
    hover = time
    toolTip = cell.noticeDescription(at: time)
    needsDisplay = true
  }
  override func mouseExited(with event: NSEvent) {
    hover = nil
    toolTip = nil
    needsDisplay = true
  }
  override func draw(_ dirtyRect: NSRect) {
    super.draw(dirtyRect)
    guard let hover, let cell = cell as? WaveformSliderCell else { return }
    let rect = cell.waveformRect
    let x = rectOrigin(hover)
    NSColor.secondaryLabelColor.setFill()
    NSRect(x: x, y: rect.minY, width: 1, height: rect.height).fill()
    let label = NSAttributedString(
      string: speechClock(hover),
      attributes: [
        .font: NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .medium),
        .foregroundColor: NSColor.labelColor,
      ])
    let width = label.size().width + 8
    let box = NSRect(
      x: min(bounds.maxX - width, max(0, x - width / 2)), y: 0, width: width, height: 14)
    NSColor.windowBackgroundColor.setFill()
    NSBezierPath(roundedRect: box, xRadius: 4, yRadius: 4).fill()
    label.draw(at: NSPoint(x: box.minX + 4, y: box.minY))
  }
  private func rectOrigin(_ time: Double) -> CGFloat {
    let rect = (cell as? WaveformSliderCell)?.waveformRect ?? bounds
    return rect.minX + time / max(1e-9, maxValue) * rect.width
  }
}

@MainActor final class WaveformSliderCell: NSSliderCell {
  var waveform: AudioWaveform?
  var notices: [StudioState.Speech.Guard] = []
  private var cachedID: UUID?
  private var cachedSize = NSSize.zero
  private var cachedAppearance: NSAppearance.Name?
  private var cachedScale: CGFloat = 0
  private var base: CGImage?, played: CGImage?
  private(set) var rebuilds = 0
  var waveformRect: NSRect {
    let native = barRect(flipped: controlView?.isFlipped ?? true)
    let bounds = controlView?.bounds ?? .zero
    // Drawing, hover and pointer seeking share one geometry. NSSlider's stock
    // private knob insets vary by OS/control size and misalign a custom track.
    let inset: CGFloat = 4
    return NSRect(
      x: native.minX + inset, y: 16, width: max(1, native.width - 2 * inset),
      height: max(1, bounds.height - 24))
  }
  var warningTrackRect: NSRect {
    let rect = waveformRect
    return NSRect(x: rect.minX, y: rect.maxY + 4, width: rect.width, height: 2)
  }
  func noticeDescription(at time: Double) -> String? {
    let descriptions = notices.filter {
      time.isFinite && $0.start.isFinite && $0.end.isFinite && $0.start <= time && time < $0.end
    }.map { notice in
      let kind = notice.dropped == true ? "Transcript omitted" : "Transcript warning"
      return "\(kind) · \(speechClock(notice.start))-\(speechClock(notice.end))\n\(notice.note)"
    }
    return descriptions.isEmpty ? nil : descriptions.joined(separator: "\n\n")
  }
  override func drawBar(inside nativeRect: NSRect, flipped: Bool) {
    let rect = waveformRect
    guard let view = controlView, let context = NSGraphicsContext.current?.cgContext else { return }
    let appearance = view.effectiveAppearance.name
    let scale = view.window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2
    if base == nil || cachedID != waveform?.id || cachedSize != rect.size
      || cachedAppearance != appearance || cachedScale != scale
    {
      cachedID = waveform?.id
      cachedSize = rect.size
      cachedAppearance = appearance
      cachedScale = scale
      base = bitmap(size: rect.size, scale: scale, color: .secondaryLabelColor)
      played = bitmap(size: rect.size, scale: scale, color: .labelColor)
      rebuilds += 1
    }
    context.saveGState()
    context.clip(to: rect)
    if let base { context.draw(base, in: rect) }
    let progress = maxValue > 0 ? min(1, max(0, doubleValue / maxValue)) : 0
    context.clip(
      to: NSRect(x: rect.minX, y: rect.minY, width: rect.width * progress, height: rect.height))
    if let played { context.draw(played, in: rect) }
    context.restoreGState()
    // Warnings describe a time span, not the played or selected region. Keep
    // them outside the neutral waveform, with a gap from the playhead as well.
    let warnings = warningTrackRect
    context.saveGState()
    context.clip(to: warnings)
    for notice in notices
    where notice.start.isFinite && notice.end.isFinite && notice.end > notice.start && maxValue > 0
    {
      let from = min(1, max(0, notice.start / maxValue))
      let to = min(1, max(0, notice.end / maxValue))
      guard to > from else { continue }
      let band = NSRect(
        x: warnings.minX + from * warnings.width, y: warnings.minY,
        width: max(2, (to - from) * warnings.width), height: warnings.height)
      (notice.dropped == true ? NSColor.systemRed : .systemOrange).withAlphaComponent(0.8)
        .setFill()
      NSBezierPath(roundedRect: band, xRadius: 1, yRadius: 1).fill()
    }
    context.restoreGState()
  }
  override func drawKnob(_ knobRect: NSRect) {
    let rect = waveformRect
    let progress = maxValue > 0 ? min(1, max(0, doubleValue / maxValue)) : 0
    NSColor.labelColor.setFill()
    NSBezierPath(
      roundedRect: NSRect(
        x: rect.minX + rect.width * progress - 1,
        y: rect.minY - 2, width: 2, height: rect.height + 4), xRadius: 1, yRadius: 1
    ).fill()
  }
  private func bitmap(size: NSSize, scale: CGFloat, color: NSColor) -> CGImage? {
    guard size.width > 0, size.height > 0,
      let context = CGContext(
        data: nil, width: max(1, Int(ceil(size.width * scale))),
        height: max(1, Int(ceil(size.height * scale))), bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { return nil }
    context.scaleBy(x: scale, y: scale)
    context.setFillColor(color.cgColor)
    let mid = size.height / 2
    guard let waveform, !waveform.minimum.isEmpty, waveform.minimum.count == waveform.maximum.count
    else {
      context.fill(CGRect(x: 0, y: mid - 0.5, width: size.width, height: 1))
      return context.makeImage()
    }
    let columns = max(1, Int(size.width))
    let count = waveform.minimum.count
    for x in 0..<columns {
      let from = x * count / columns
      let to = min(count, max(from + 1, (x + 1) * count / columns))
      var low: Float = 0
      var high: Float = 0
      for i in from..<to {
        low = min(low, waveform.minimum[i])
        high = max(high, waveform.maximum[i])
      }
      let lower = max(0.5, min(1, -CGFloat(low * waveform.displayGain)) * (mid - 1))
      let upper = max(0.5, min(1, CGFloat(high * waveform.displayGain)) * (mid - 1))
      context.fill(CGRect(x: CGFloat(x), y: mid - lower, width: 1, height: lower + upper))
    }
    return context.makeImage()
  }
}
