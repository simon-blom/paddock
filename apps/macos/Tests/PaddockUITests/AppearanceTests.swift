import AppKit
import PaddockDesign
import Testing

struct AppearanceTests {
  @Test func dynamicAppKitPaletteUsesTheSameOpaqueColors() throws {
    for role in ["canvas", "surface", "popup"] {
      let dynamic = PaddockAppearance.nsColor(role)
      for dark in [false, true, false] {
        let appearance = try #require(NSAppearance(named: dark ? .darkAqua : .aqua))
        var resolved: NSColor?
        appearance.performAsCurrentDrawingAppearance { resolved = dynamic.usingColorSpace(.sRGB) }
        let color = try #require(resolved)
        let expected = PaddockAppearance.nsColor(role, dark: dark)
        #expect(abs(color.redComponent - expected.redComponent) < 0.001)
        #expect(abs(color.greenComponent - expected.greenComponent) < 0.001)
        #expect(abs(color.blueComponent - expected.blueComponent) < 0.001)
        #expect(color.alphaComponent == 1)
      }
    }
  }
  @Test func bundledPaletteHasConsistentNativeRolesAndRadii() {
    for dark in [false, true] {
      for role in [
        "canvas", "sidebar", "surface", "popup", "elevated", "primary", "secondary", "border",
        "accent",
      ] {
        let color = PaddockAppearance.nsColor(role, dark: dark)
        #expect(color.redComponent == color.greenComponent)
        #expect(color.greenComponent == color.blueComponent)
        #expect(color.alphaComponent == 1)
      }
      let warning = PaddockAppearance.nsColor("caution", dark: dark)
      #expect(warning.redComponent != warning.blueComponent)
    }
    #expect(PaddockAppearance.Radius.control == 6)
    #expect(PaddockAppearance.Radius.card == 8)
    #expect(PaddockAppearance.Radius.panel == 12)
  }

  @Test func secondaryTextRemainsReadableOnChromeSurfaces() {
    func luminance(_ color: NSColor) -> CGFloat {
      func channel(_ value: CGFloat) -> CGFloat {
        value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4)
      }
      return channel(color.redComponent) * 0.2126 + channel(color.greenComponent) * 0.7152
        + channel(color.blueComponent) * 0.0722
    }
    for dark in [false, true] {
      for background in ["canvas", "surface", "popup", "elevated"] {
        let fg = luminance(PaddockAppearance.nsColor("secondary", dark: dark))
        let bg = luminance(PaddockAppearance.nsColor(background, dark: dark))
        #expect((max(fg, bg) + 0.05) / (min(fg, bg) + 0.05) >= 4.5)
      }
    }
  }
}
