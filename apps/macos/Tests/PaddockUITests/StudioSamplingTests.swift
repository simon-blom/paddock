import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native sampling controls", .serialized) @MainActor
struct StudioSamplingTests {
  @Test func trackingSnapsWithoutTickMarks() throws {
    let slider = SamplingSliderControl()
    slider.minValue = 0.01
    slider.maxValue = 1
    slider.valueStep = 0.01
    var changes: [Double] = []
    slider.onChange = { changes.append($0) }
    slider.doubleValue = 0.95
    #expect(changes.isEmpty)  // installing an advertised default isn't an edit
    #expect(slider.numberOfTickMarks == 0)
    #expect(slider.cell is SamplingSliderCell)
    #expect(!slider.allowsTickMarkValuesOnly)
    #expect(slider.isContinuous)
    slider.doubleValue = 0.537
    slider.sendAction(slider.action, to: slider.target)
    #expect(abs(slider.doubleValue - 0.54) < 0.000001)
    #expect(changes.count == 1)
    slider.commit(-10)
    #expect(slider.doubleValue == 0.01)
    slider.commit(10)
    #expect(slider.doubleValue == 1)
    slider.commit(.nan)
    #expect(changes.count == 3)
  }

  @Test func keyboardAndAccessibilityUseTheSameDiscreteSteps() throws {
    let slider = SamplingSliderControl()
    #expect(slider.acceptsFirstResponder)
    slider.minValue = 0
    slider.maxValue = 2
    slider.valueStep = 0.05
    slider.doubleValue = 1
    var changes = 0
    slider.onChange = { _ in changes += 1 }
    func key(_ code: UInt16) throws {
      let event = try #require(
        NSEvent.keyEvent(
          with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0,
          windowNumber: 0, context: nil, characters: "", charactersIgnoringModifiers: "",
          isARepeat: false, keyCode: code))
      slider.keyDown(with: event)
    }
    try key(124)
    #expect(abs(slider.doubleValue - 1.05) < 0.000001)
    #expect(slider.accessibilityPerformIncrement())
    #expect(abs(slider.doubleValue - 1.10) < 0.000001)
    #expect(slider.accessibilityPerformDecrement())
    #expect(abs(slider.doubleValue - 1.05) < 0.000001)
    try key(123)
    #expect(slider.doubleValue == 1)
    try key(115)
    #expect(slider.doubleValue == 0)
    try key(119)
    #expect(slider.doubleValue == 2)
    slider.setAccessibilityValue(NSNumber(value: 1.126))
    #expect(abs(slider.doubleValue - 1.15) < 0.000001)
    let before = changes
    slider.isEnabled = false
    #expect(!slider.acceptsFirstResponder)
    try key(124)
    #expect(!slider.accessibilityPerformIncrement())
    slider.setAccessibilityValue(NSNumber(value: 0))
    #expect(changes == before)
  }

  @Test func populatedRowsFitBothAppearancesWithoutRulerTracks() async throws {
    let json = """
      [
        {"key":"temperature","label":"Temperature","min":0,"max":2,"step":0.05,"value":1,"display":"Default (1.0)","set":false},
        {"key":"topP","label":"Top-p","min":0.01,"max":1,"step":0.01,"value":0.95,"display":"Default (0.95)","set":false},
        {"key":"topK","label":"Top-k","min":0,"max":200,"step":1,"value":20,"display":"Default (20)","set":false},
        {"key":"minP","label":"Min-p","min":0,"max":1,"step":0.01,"value":0,"display":"Default (off)","set":false},
        {"key":"presencePenalty","label":"Presence penalty","min":-2,"max":2,"step":0.1,"value":0,"display":"Default (off)","set":false},
        {"key":"repeatPenalty","label":"Repeat penalty","min":1,"max":2,"step":0.01,"value":1,"display":"Default (off)","set":false}
      ]
      """
    let dials = try JSONDecoder().decode([StudioState.Composer.Dial].self, from: Data(json.utf8))
    for dark in [false, true] {
      let host = NSHostingController(
        rootView:
          VStack(spacing: 12) {
            ForEach(dials) { dial in
              StudioSamplingRow(dial: dial, value: .constant(dial.value), display: dial.display)
            }
          }.padding(16).frame(width: 320).studioPopoverSurface()
          .environment(\.colorScheme, dark ? .dark : .light))
      let size = host.sizeThatFits(in: CGSize(width: 320, height: 600))
      #expect(size.width == 320)
      #expect(size.height > 250 && size.height < 400)
      host.view.frame = NSRect(origin: .zero, size: size)
      host.view.layoutSubtreeIfNeeded()
      func sliders(in view: NSView) -> [SamplingSliderControl] {
        (view as? SamplingSliderControl).map { [$0] } ?? view.subviews.flatMap { sliders(in: $0) }
      }
      let controls = sliders(in: host.view)
      #expect(controls.count == 6)
      for control in controls {
        #expect(control.numberOfTickMarks == 0)
        #expect(control.frame.height == 18)
        #expect(control.frame.width >= 280 && control.frame.width <= 288)
        #expect(control.accessibilityLabel()?.isEmpty == false)
      }
    }
  }
}
