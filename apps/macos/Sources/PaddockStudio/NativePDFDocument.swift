import CoreGraphics
import Foundation
import ImageIO
import PaddockConversationCore
import UniformTypeIdentifiers

extension NativePDFInput {
  func source(_ bytes: Data) throws -> NativeDocumentSource {
    let document = try NativePDFDocument(bytes)
    return NativeDocumentSource(pageCount: document.pageCount) { number, maxPixels in
      try await document.render(number, maxPixels: maxPixels)
    }
  }
}

/// One compressed original retained by CGPDFDocument. No raster cache and no
/// AppKit/WebKit dependency; each page's bitmap is released before the next.
actor NativePDFDocument {
  nonisolated let pageCount: Int
  private let document: CGPDFDocument
  init(_ bytes: Data) throws {
    guard let provider = CGDataProvider(data: bytes as CFData), let doc = CGPDFDocument(provider),
      !doc.isEncrypted || doc.isUnlocked, doc.numberOfPages > 0
    else { throw ConversationFailure.invalid("The PDF is damaged or password protected") }
    document = doc
    pageCount = doc.numberOfPages
  }
  func render(_ number: Int, maxPixels: Int) throws -> NativeRasterPage {
    try Task.checkCancellation()
    guard (1...pageCount).contains(number), (1...32 * 1024 * 1024).contains(maxPixels),
      let page = document.page(at: number)
    else { throw ConversationFailure.invalid("Invalid PDF page or pixel budget") }
    return try autoreleasepool {
      let image = try Self.raster(page, maxPixels: maxPixels)
      let data = NSMutableData()
      guard
        let output = CGImageDestinationCreateWithData(
          data, UTType.jpeg.identifier as CFString, 1, nil)
      else {
        throw ConversationFailure.invalid("PDF image encoding failed")
      }
      CGImageDestinationAddImage(
        output, image, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
      guard CGImageDestinationFinalize(output) else {
        throw ConversationFailure.invalid("PDF image encoding failed")
      }
      return NativeRasterPage(
        image: [
          "type": .string("input_image"), "detail": .string("auto"),
          "image_url": .string("data:image/jpeg;base64,\((data as Data).base64EncodedString())"),
        ], width: image.width, height: image.height)
    }
  }

  /// Shared geometry for model input and native figure previews, including
  /// crop-box origin and rotation. Call only off the main actor.
  nonisolated static func raster(_ page: CGPDFPage, maxPixels: Int) throws -> CGImage {
    guard (1...32 * 1024 * 1024).contains(maxPixels) else {
      throw ConversationFailure.invalid("Invalid PDF pixel budget")
    }
    let rect = page.getBoxRect(.cropBox)
    guard rect.origin.x.isFinite, rect.origin.y.isFinite,
      rect.width.isFinite, rect.height.isFinite, rect.width > 0, rect.height > 0
    else {
      throw ConversationFailure.invalid("Invalid PDF page geometry")
    }
    let rotated = abs(page.rotationAngle) % 180 == 90
    let width = rotated ? rect.height : rect.width
    let height = rotated ? rect.width : rect.height
    guard (width * height).isFinite, width * height > 0 else {
      throw ConversationFailure.invalid("Invalid PDF page area")
    }
    let scale = min(sqrt(Double(maxPixels) / (width * height)), 16384 / max(width, height))
    var w = max(1, Int(floor(width * scale)))
    var h = max(1, Int(floor(height * scale)))
    // A very thin page can round its short side up to one pixel. Keep the
    // area budget hard even then; the drawing transform preserves aspect.
    if w * h > maxPixels {
      if w >= h { w = max(1, maxPixels / h) } else { h = max(1, maxPixels / w) }
    }
    guard
      let context = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw ConversationFailure.invalid("Not enough memory to render PDF page") }
    let destination = CGRect(x: 0, y: 0, width: w, height: h)
    context.setFillColor(CGColor(gray: 1, alpha: 1))
    context.fill(destination)
    context.concatenate(
      page.getDrawingTransform(.cropBox, rect: destination, rotate: 0, preserveAspectRatio: true))
    context.drawPDFPage(page)
    try Task.checkCancellation()
    guard let image = context.makeImage() else {
      throw ConversationFailure.invalid("PDF rasterization failed")
    }
    return image
  }
}
