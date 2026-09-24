// Renders Paddock.icns, the macOS app icon, from the same Truespar mark the
// Studio shows (studio/public/img/truespar-mark-3d.svg) - one source for the
// art, as with assets/paddock.ico. The mark sits on a plate that follows
// Apple's macOS icon grid (824 of 1024, continuous corners), because macOS 26
// shrinks an icon without that shape onto a grey plate of its own.
//
//   swift packaging/macos/make-icon.swift
//
// Rerun it when the mark changes and commit the .icns next to it. The build
// scripts only copy the file; nothing renders at build time.
import AppKit
import SwiftUI

let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let root = here.deletingLastPathComponent().deletingLastPathComponent()
let markURL = root.appendingPathComponent("studio/public/img/truespar-mark-3d.svg")
guard let mark = NSImage(contentsOf: markURL) else {
  fatalError("Cannot read the mark at \(markURL.path)")
}

// Apple's template: a 1024 canvas, an 824 plate at 100/100, radius 185.4.
let plate = CGRect(x: 100, y: 100, width: 824, height: 824)
let plateShape = RoundedRectangle(cornerRadius: 185.4, style: .continuous).path(in: plate).cgPath
let plateTop = NSColor(srgbRed: 0x18 / 255, green: 0x3E / 255, blue: 0x5C / 255, alpha: 1).cgColor
let plateBottom = NSColor(srgbRed: 0x09 / 255, green: 0x1A / 255, blue: 0x2B / 255, alpha: 1).cgColor
// The SVG's viewBox is centred on the mark, whose extent is 1068 units high.
let markHeight: CGFloat = 540
let markSide = 1228 * markHeight / 1068

func render(pixels: Int) -> Data {
  let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8,
    samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
    bytesPerRow: 0, bitsPerPixel: 0)!
  let context = NSGraphicsContext(bitmapImageRep: rep)!
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = context
  let cg = context.cgContext
  let scale = CGFloat(pixels) / 1024
  cg.scaleBy(x: scale, y: scale)
  // Shadows are in device space, so their geometry is scaled by hand.
  func shadow(_ y: CGFloat, _ blur: CGFloat, _ alpha: CGFloat) {
    cg.setShadow(
      offset: CGSize(width: 0, height: -y * scale), blur: blur * scale,
      color: NSColor(white: 0, alpha: alpha).cgColor)
  }

  cg.saveGState()
  shadow(10, 20, 0.3)
  cg.addPath(plateShape)
  cg.setFillColor(plateBottom)
  cg.fillPath()
  cg.restoreGState()

  cg.saveGState()
  cg.addPath(plateShape)
  cg.clip()
  let gradient = CGGradient(
    colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: [plateTop, plateBottom] as CFArray,
    locations: [0, 1])!
  cg.drawLinearGradient(
    gradient, start: CGPoint(x: 512, y: plate.maxY), end: CGPoint(x: 512, y: plate.minY), options: [])
  cg.restoreGState()

  // One transparency layer, so the mark casts a single shadow rather than
  // one per face.
  cg.saveGState()
  shadow(14, 28, 0.35)
  cg.beginTransparencyLayer(auxiliaryInfo: nil)
  mark.draw(in: NSRect(x: 512 - markSide / 2, y: 512 - markSide / 2, width: markSide, height: markSide))
  cg.endTransparencyLayer()
  cg.restoreGState()

  NSGraphicsContext.restoreGraphicsState()
  return rep.representation(using: .png, properties: [:])!
}

let iconset = FileManager.default.temporaryDirectory
  .appendingPathComponent("Paddock-\(UUID().uuidString).iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: iconset) }
for points in [16, 32, 128, 256, 512] {
  try render(pixels: points).write(to: iconset.appendingPathComponent("icon_\(points)x\(points).png"))
  try render(pixels: points * 2).write(
    to: iconset.appendingPathComponent("icon_\(points)x\(points)@2x.png"))
}
let output = here.appendingPathComponent("Paddock.icns")
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["--convert", "icns", "--output", output.path, iconset.path]
try iconutil.run()
iconutil.waitUntilExit()
guard iconutil.terminationStatus == 0 else { fatalError("iconutil failed") }
print("Wrote \(output.path)")
