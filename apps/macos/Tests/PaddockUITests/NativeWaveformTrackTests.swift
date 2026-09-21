import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native waveform interaction", .serialized) @MainActor
struct NativeWaveformTrackTests {
  @Test func keyboardAndAccessibilityUseARealSlider() throws {
    _ = NSApplication.shared
    var seeks: [Double] = []
    var toggles = 0
    let host = NSHostingView(
      rootView: NativeWaveformTrack(
        waveform: fixture(), duration: 10, position: 3,
        onSeek: { seeks.append($0) }, onToggle: { toggles += 1 }
      ).frame(width: 400, height: 48))
    host.frame = NSRect(x: 0, y: 0, width: 400, height: 48)
    host.layoutSubtreeIfNeeded()
    let slider = try #require(find(host))
    #expect(slider.accessibilityRole() == .slider)
    #expect(slider.accessibilityChildren()?.isEmpty == true)
    #expect(slider.accessibilityLabel() == "Recording position")
    #expect((slider.accessibilityValue() as? NSNumber)?.doubleValue == 3)
    #expect((slider.accessibilityMinValue() as? NSNumber)?.doubleValue == 0)
    #expect((slider.accessibilityMaxValue() as? NSNumber)?.doubleValue == 10)
    #expect(slider.isContinuous && slider.isEnabled)
    func key(_ code: UInt16) throws {
      let event = try #require(
        NSEvent.keyEvent(
          with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0,
          windowNumber: 0, context: nil, characters: "", charactersIgnoringModifiers: "",
          isARepeat: false, keyCode: code))
      slider.keyDown(with: event)
    }
    try key(124)
    try key(124)
    try key(123)
    try key(115)
    try key(119)
    try key(49)
    #expect(seeks == [8, 10, 5, 0, 10])
    #expect(toggles == 1)
    #expect(slider.accessibilityPerformDecrement())
    #expect(slider.doubleValue == 5)
    #expect(slider.accessibilityPerformIncrement())
    slider.setAccessibilityValue(NSNumber(value: 2.25))
    #expect(slider.doubleValue == 2.25 && seeks.last == 2.25)
  }
  @Test func unknownLengthDoesNotInventASeekRange() throws {
    let host = NSHostingView(
      rootView: NativeWaveformTrack(
        waveform: nil, duration: 0, position: 0, onSeek: { _ in }, onToggle: {}))
    host.frame = NSRect(x: 0, y: 0, width: 300, height: 48)
    host.layoutSubtreeIfNeeded()
    #expect(try !#require(find(host)).isEnabled)
  }
  @Test func renderingCachesEnvelopeAcrossPlaybackAndRebuildsOnResizeAndTheme() throws {
    _ = NSApplication.shared
    let notices = try JSONDecoder().decode(
      [StudioState.Speech.Guard].self,
      from: Data(
        #"[{"start":2,"end":3,"note":"Cut at the limit","dropped":false},{"start":7,"end":9,"note":"No speech","dropped":true}]"#
          .utf8))
    let host = NSHostingView(
      rootView: NativeWaveformTrack(
        waveform: fixture(), duration: 10, position: 0,
        notices: notices, onSeek: { _ in }, onToggle: {}
      ).frame(width: 600, height: 48))
    host.frame = NSRect(x: 0, y: 0, width: 600, height: 48)
    host.layoutSubtreeIfNeeded()
    let slider = try #require(find(host))
    let cell = try #require(slider.cell as? WaveformSliderCell)
    #expect(
      slider.bounds.height == 48,
      "NSSlider's default 16-point intrinsic height would clip the waveform")
    #expect(cell.warningTrackRect.minY > cell.waveformRect.maxY)
    #expect(cell.warningTrackRect.maxY <= slider.bounds.maxY)
    func draw(_ name: String) throws -> Data {
      let image = try #require(slider.bitmapImageRepForCachingDisplay(in: slider.bounds))
      slider.cacheDisplay(in: slider.bounds, to: image)
      var coloredRows = Set<Int>()
      for y in 0..<image.pixelsHigh {
        for x in 0..<image.pixelsWide {
          guard let color = image.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
            color.alphaComponent > 0.05
          else { continue }
          let channels = [color.redComponent, color.greenComponent, color.blueComponent]
          if channels.max()! - channels.min()! > 0.1 { coloredRows.insert(y) }
        }
      }
      #expect(!coloredRows.isEmpty, "Warnings must remain visible")
      let scale = CGFloat(image.pixelsHigh) / slider.bounds.height
      #expect(
        coloredRows.count <= Int(ceil(3 * scale)), "Warnings must not tint the waveform background")
      let png = try #require(image.representation(using: .png, properties: [:]))
      if let directory = ProcessInfo.processInfo.environment["PADDOCK_WAVEFORM_CAPTURE"] {
        try png.write(to: URL(fileURLWithPath: directory).appendingPathComponent(name + ".png"))
      }
      return png
    }
    slider.appearance = NSAppearance(named: .aqua)
    let start = try draw("light-start")
    let count = cell.rebuilds
    slider.doubleValue = 5
    let middle = try draw("light-middle")
    slider.doubleValue = 10
    let end = try draw("light-end")
    #expect(start != middle && middle != end)
    #expect(cell.rebuilds == count, "Playback must not re-decode or rebuild peak bitmaps")
    #expect(cell.waveformRect.width > 500)
    slider.appearance = NSAppearance(named: .darkAqua)
    slider.doubleValue = 5
    _ = try draw("dark-middle")
    #expect(cell.rebuilds == count + 1)
    slider.setFrameSize(NSSize(width: 250, height: 48))
    _ = try draw("dark-narrow")
    #expect(cell.rebuilds == count + 2)
    let previousAppearance = slider.effectiveAppearance.name
    slider.appearance = NSAppearance(named: .accessibilityHighContrastDarkAqua)
    let expectedRebuilds =
      count + 2 + (slider.effectiveAppearance.name != previousAppearance ? 1 : 0)
    _ = try draw("contrast-narrow")
    #expect(cell.rebuilds == expectedRebuilds)
    let bitmap = try #require(slider.bitmapImageRepForCachingDisplay(in: slider.bounds))
    var milliseconds: [Double] = []
    for frame in 0..<300 {
      slider.doubleValue = Double(frame) / 30
      let start = ContinuousClock.now
      slider.cacheDisplay(in: slider.bounds, to: bitmap)
      let time = start.duration(to: .now).components
      milliseconds.append(Double(time.seconds) * 1000 + Double(time.attoseconds) / 1e15)
    }
    milliseconds.sort()
    #expect(cell.rebuilds == expectedRebuilds)
    print(
      "WAVEFORM_DRAW_BENCH cached offscreen draw: p50 \(milliseconds[150]) ms; p99 \(milliseconds[297]) ms"
    )
  }
  @Test func pointerTrackingUsesTheWholeWaveformAndHoverExplainsWarnings() throws {
    _ = NSApplication.shared
    var seeks: [Double] = []
    let notices = try JSONDecoder().decode(
      [StudioState.Speech.Guard].self,
      from: Data(#"[{"start":2,"end":4,"note":"Audio was cut"}]"#.utf8))
    let host = NSHostingView(
      rootView: NativeWaveformTrack(
        waveform: fixture(), duration: 10, position: 0,
        notices: notices, onSeek: { seeks.append($0) }, onToggle: {}
      ).frame(width: 400, height: 48))
    // An unshown window supplies real AppKit coordinates without screen flicker.
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 400, height: 48),
      styleMask: .borderless, backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    defer { window.close() }
    window.contentView = host
    host.layoutSubtreeIfNeeded()
    let slider = try #require(find(host))
    let cell = try #require(slider.cell as? WaveformSliderCell)
    let image = try #require(slider.bitmapImageRepForCachingDisplay(in: slider.bounds))
    slider.cacheDisplay(in: slider.bounds, to: image)
    let bar = cell.waveformRect
    func event(_ kind: NSEvent.EventType, _ fraction: Double) throws -> NSEvent {
      let point = slider.convert(NSPoint(x: bar.minX + fraction * bar.width, y: bar.midY), to: nil)
      return try #require(
        NSEvent.mouseEvent(
          with: kind, location: point, modifierFlags: [],
          timestamp: 0, windowNumber: window.windowNumber, context: nil, eventNumber: 0,
          clickCount: 1, pressure: 1))
    }
    slider.mouseMoved(with: try event(.mouseMoved, 0.3))
    #expect(slider.toolTip == "Transcript warning · 0:02-0:04\nAudio was cut")
    slider.setAccessibilityValue(NSNumber(value: 3))
    #expect(slider.accessibilityHelp()?.contains("Transcript warning · 0:02-0:04") == true)
    for fraction in [0.25, 0.5, 0.75] {
      slider.mouseDown(with: try event(.leftMouseDown, fraction))
      #expect(slider.tracking)
      slider.mouseUp(with: try event(.leftMouseUp, fraction))
      #expect(!slider.tracking)
      #expect(abs(slider.doubleValue - fraction * 10) < 0.01)
    }
    #expect(!seeks.isEmpty)
    slider.mouseDown(with: try event(.leftMouseDown, 0.2))
    slider.mouseDragged(with: try event(.leftMouseDragged, 0.8))
    #expect(slider.doubleValue == 8)
    slider.mouseDragged(with: try event(.leftMouseDragged, 1.5))
    #expect(slider.doubleValue == 10)
    slider.mouseUp(with: try event(.leftMouseUp, -0.5))
    #expect(slider.doubleValue == 0 && !slider.tracking)
    slider.mouseExited(with: try event(.mouseMoved, 0.9))
    #expect(slider.toolTip == nil)
  }
  private func fixture() -> AudioWaveform {
    let amplitudes: [Float] = (0..<2048).map { i in
      let x = Double(i) / 2048
      return (x > 0.3 && x < 0.45) ? 0 : Float(abs(sin(x * 50)) * 0.75 + 0.05)
    }
    return AudioWaveform(minimum: amplitudes.map { -$0 }, maximum: amplitudes, duration: 10)
  }
  private func find(_ view: NSView) -> WaveformSlider? {
    (view as? WaveformSlider) ?? view.subviews.lazy.compactMap(find).first
  }
}
