import AppKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Neutral workspace palette") @MainActor
struct PaletteTests {
  @Test func chromeRemainsAchromaticAndOpaqueInBothAppearances() throws {
    for appearance in [NSAppearance.Name.aqua, .darkAqua] {
      for color in [
        PaddockStyle.canvas, PaddockStyle.sidebar, PaddockStyle.surface,
        PaddockStyle.elevated, PaddockStyle.accent, PaddockStyle.actionForeground,
        PaddockStyle.border,
      ] {
        let resolved = try resolve(color, appearance: appearance)
        #expect(abs(resolved.redComponent - resolved.greenComponent) < 0.0001)
        #expect(abs(resolved.greenComponent - resolved.blueComponent) < 0.0001)
        #expect(resolved.alphaComponent == 1)
      }
    }
  }

  @Test func enabledActionAndWarningTextHaveStrongContrast() throws {
    for appearance in [NSAppearance.Name.aqua, .darkAqua] {
      let ink = try resolve(PaddockStyle.actionForeground, appearance: appearance)
      let action = try resolve(PaddockStyle.accent, appearance: appearance)
      #expect(contrast(ink, action) >= 7)
      let warning = try resolve(PaddockStyle.caution, appearance: appearance)
      for background in [PaddockStyle.canvas, PaddockStyle.surface] {
        #expect(contrast(warning, try resolve(background, appearance: appearance)) >= 4.5)
      }
    }
  }

  private func resolve(_ color: Color, appearance: NSAppearance.Name) throws -> NSColor {
    let appearance = try #require(NSAppearance(named: appearance))
    var resolved: NSColor?
    appearance.performAsCurrentDrawingAppearance {
      resolved = NSColor(color).usingColorSpace(.sRGB)
    }
    return try #require(resolved)
  }

  private func contrast(_ first: NSColor, _ second: NSColor) -> CGFloat {
    let a = luminance(first)
    let b = luminance(second)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)
  }

  private func luminance(_ color: NSColor) -> CGFloat {
    func linear(_ component: CGFloat) -> CGFloat {
      component <= 0.04045 ? component / 12.92 : pow((component + 0.055) / 1.055, 2.4)
    }
    return 0.2126 * linear(color.redComponent) + 0.7152 * linear(color.greenComponent)
      + 0.0722 * linear(color.blueComponent)
  }
}
