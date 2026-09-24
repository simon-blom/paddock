// Renders the Paddock icons from one piece of art, the mark in
// studio/public/img/paddock-mark.svg (the Studio shows the same file):
//
//   packaging/macos/Paddock.icns  the macOS app icon, on Apple's icon grid
//   assets/paddock.ico            the two exes and the Studio's favicon, with
//                                 the plate to the edge and no macOS margin
//
// Each alternative mark under assets/icon-alternatives/<name>/paddock-mark.svg
// gets the same pair rendered next to it. Alternatives are not wired in
// anywhere; they are kept so a switch is one file move, not a redraw.
//
// The plate follows Apple's macOS icon grid (824 of 1024, continuous corners),
// because macOS 26 shrinks an icon without that shape onto a grey plate of its
// own. The mark SVGs use the same 1024 grid, so they are placed as drawn.
//
//   swift packaging/macos/make-icon.swift
//
// Rerun it when a mark changes and commit the results. The build scripts only
// copy the files; nothing renders at build time.
import AppKit
import SwiftUI

let here = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let root = here.deletingLastPathComponent().deletingLastPathComponent()

// Apple's template: a 1024 canvas, an 824 plate at 100/100, radius 185.4.
let plate = CGRect(x: 100, y: 100, width: 824, height: 824)
let plateShape = RoundedRectangle(cornerRadius: 185.4, style: .continuous).path(in: plate).cgPath
// The marks' viewBox, "232 242 560 560", flipped into CoreGraphics' y-up space.
let markRect = CGRect(x: 232, y: 1024 - 242 - 560, width: 560, height: 560)

func srgb(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat = 1) -> CGColor {
  NSColor(srgbRed: r / 255, green: g / 255, blue: b / 255, alpha: a).cgColor
}
let space = CGColorSpace(name: CGColorSpace.sRGB)!
let plateFill = CGGradient(
  colorsSpace: space, colors: [srgb(39, 43, 51), srgb(11, 12, 15)] as CFArray, locations: [0, 1])!
// A soft sheen over the top of the plate, gone by 45 % of its height.
let sheen = CGGradient(
  colorsSpace: space, colors: [srgb(255, 255, 255, 0.18), srgb(255, 255, 255, 0)] as CFArray,
  locations: [0, 1])!

/// Plate and mark in a `pixels` square. On the macOS grid the plate keeps its
/// 100-unit margin and casts a shadow; otherwise the plate fills the square.
func render(_ mark: NSImage, pixels: Int, macGrid: Bool) -> NSBitmapImageRep {
  let rep = NSBitmapImageRep(
    bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels, bitsPerSample: 8,
    samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
    bytesPerRow: 0, bitsPerPixel: 0)!
  let context = NSGraphicsContext(bitmapImageRep: rep)!
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = context
  let cg = context.cgContext
  let scale = CGFloat(pixels) / (macGrid ? 1024 : plate.width)
  cg.scaleBy(x: scale, y: scale)
  if !macGrid { cg.translateBy(x: -plate.minX, y: -plate.minY) }

  if macGrid {
    // Shadows are in device space, so their geometry is scaled by hand.
    cg.saveGState()
    cg.setShadow(
      offset: CGSize(width: 0, height: -14 * scale), blur: 30 * scale,
      color: NSColor(white: 0, alpha: 0.28).cgColor)
    cg.addPath(plateShape)
    cg.setFillColor(srgb(11, 12, 15))
    cg.fillPath()
    cg.restoreGState()
  }

  cg.saveGState()
  cg.addPath(plateShape)
  cg.clip()
  cg.drawLinearGradient(
    plateFill, start: CGPoint(x: 512, y: plate.maxY), end: CGPoint(x: 512, y: plate.minY),
    options: [])
  mark.draw(in: markRect)
  cg.drawLinearGradient(
    sheen, start: CGPoint(x: 512, y: plate.maxY),
    end: CGPoint(x: 512, y: plate.maxY - 0.45 * plate.height), options: [])
  cg.restoreGState()

  NSGraphicsContext.restoreGraphicsState()
  return rep
}

func png(_ rep: NSBitmapImageRep) -> Data { rep.representation(using: .png, properties: [:])! }

func writeIcns(_ mark: NSImage, to output: URL) throws {
  let iconset = FileManager.default.temporaryDirectory
    .appendingPathComponent("Paddock-\(UUID().uuidString).iconset")
  try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
  defer { try? FileManager.default.removeItem(at: iconset) }
  for points in [16, 32, 128, 256, 512] {
    try png(render(mark, pixels: points, macGrid: true))
      .write(to: iconset.appendingPathComponent("icon_\(points)x\(points).png"))
    try png(render(mark, pixels: points * 2, macGrid: true))
      .write(to: iconset.appendingPathComponent("icon_\(points)x\(points)@2x.png"))
  }
  let iconutil = Process()
  iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
  iconutil.arguments = ["--convert", "icns", "--output", output.path, iconset.path]
  try iconutil.run()
  iconutil.waitUntilExit()
  guard iconutil.terminationStatus == 0 else { fatalError("iconutil failed") }
}

/// The sizes the exes and the favicon have always carried. 256 is stored as
/// PNG, which the Studio slices out as its apple-touch-icon; the rest are
/// 32-bit BMPs, which every Windows shell reads.
func writeIco(_ mark: NSImage, to output: URL) throws {
  var images: [(Int, Data)] = []
  for pixels in [256, 128, 64, 48, 40, 32, 24, 20, 16] {
    let rep = render(mark, pixels: pixels, macGrid: false)
    images.append((pixels, pixels >= 256 ? png(rep) : dib(rep)))
  }
  var ico = Data()
  func u16(_ v: Int) { ico.append(contentsOf: [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)]) }
  func u32(_ v: Int) { u16(v & 0xFFFF); u16(v >> 16 & 0xFFFF) }
  u16(0); u16(1); u16(images.count)
  var offset = 6 + 16 * images.count
  for (pixels, data) in images {
    // 0 in the width and height bytes means 256; the format cannot say it.
    ico.append(contentsOf: [UInt8(pixels & 0xFF), UInt8(pixels & 0xFF), 0, 0])
    u16(1); u16(32); u32(data.count); u32(offset)
    offset += data.count
  }
  for (_, data) in images { ico.append(data) }
  try ico.write(to: output)
}

/// A 32-bit BMP body as an .ico wants it: header with a doubled height,
/// straight-alpha BGRA rows bottom-up, then the AND mask with a set bit for
/// every clear pixel. Windows reads the alpha, but readers that honour the
/// mask instead paint the corners opaque without it.
func dib(_ rep: NSBitmapImageRep) -> Data {
  let w = rep.pixelsWide, h = rep.pixelsHigh
  let maskRow = (w + 31) / 32 * 4
  var out = Data()
  var mask = Data(count: maskRow * h)
  func u16(_ v: Int) { out.append(contentsOf: [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)]) }
  func u32(_ v: Int) { u16(v & 0xFFFF); u16(v >> 16 & 0xFFFF) }
  u32(40); u32(w); u32(h * 2); u16(1); u16(32); u32(0); u32(w * h * 4)
  u32(0); u32(0); u32(0); u32(0)
  let bytes = rep.bitmapData!
  for (row, y) in (0..<h).reversed().enumerated() {
    for x in 0..<w {
      let p = bytes + y * rep.bytesPerRow + x * 4
      let a = Int(p[3])
      // The bitmap is premultiplied; an .ico wants straight alpha.
      func straight(_ c: UInt8) -> UInt8 { a == 0 ? 0 : UInt8(min(255, Int(c) * 255 / a)) }
      out.append(contentsOf: [straight(p[2]), straight(p[1]), straight(p[0]), UInt8(a)])
      if a == 0 { mask[row * maskRow + x / 8] |= 0x80 >> UInt8(x % 8) }
    }
  }
  out.append(mask)
  return out
}

func mark(_ url: URL) -> NSImage {
  guard let image = NSImage(contentsOf: url) else { fatalError("Cannot read the mark at \(url.path)") }
  return image
}

let main = mark(root.appendingPathComponent("studio/public/img/paddock-mark.svg"))
try writeIcns(main, to: here.appendingPathComponent("Paddock.icns"))
try writeIco(main, to: root.appendingPathComponent("assets/paddock.ico"))
print("Wrote packaging/macos/Paddock.icns and assets/paddock.ico")

let alternatives = root.appendingPathComponent("assets/icon-alternatives")
for folder in (try? FileManager.default.contentsOfDirectory(
  at: alternatives, includingPropertiesForKeys: nil)) ?? []
where FileManager.default.fileExists(atPath: folder.appendingPathComponent("paddock-mark.svg").path) {
  let alternative = mark(folder.appendingPathComponent("paddock-mark.svg"))
  try writeIcns(alternative, to: folder.appendingPathComponent("Paddock.icns"))
  try writeIco(alternative, to: folder.appendingPathComponent("paddock.ico"))
  print("Wrote the \(folder.lastPathComponent) alternative")
}
