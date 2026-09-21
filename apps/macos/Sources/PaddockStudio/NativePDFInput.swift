import CoreGraphics
import Foundation
import ImageIO
import PaddockClient
import PaddockConversationCore
import UniformTypeIdentifiers

/// Metal runners without PDFium still receive selected PDF page pixels.
/// CoreGraphics does this natively, off-main, one page at a time. It never
/// creates a Lector view, browser canvas or a second uploaded original.
actor NativePDFInput {
  static let shared = NativePDFInput()
  func parts(_ bytes: Data, metadata: [String: ConversationValue]) throws -> [ConversationValue] {
    typealias V = ConversationValue
    guard let provider = CGDataProvider(data: bytes as CFData),
      let document = CGPDFDocument(provider), !document.isEncrypted || document.isUnlocked
    else {
      throw ManagerError.core("The PDF is damaged or password protected")
    }
    var first = 1
    var last = document.numberOfPages
    if let range = metadata["pageRange"]?.string {
      let bounds = range.split(separator: "-", omittingEmptySubsequences: false)
      first = bounds.first.flatMap { Int($0) } ?? 1
      if bounds.count > 1 { last = Int(bounds[1]) ?? last } else { last = first }
    }
    guard first >= 1, last >= first, last <= document.numberOfPages else {
      throw ManagerError.core("The PDF page range is invalid")
    }
    guard last - first < 64 else { throw ManagerError.core("Select up to 64 PDF pages per turn") }
    var result: [V] = []
    for index in first...last {
      try Task.checkCancellation()
      let encoded: Data = try autoreleasepool {
        guard let page = document.page(at: index) else {
          throw ManagerError.core("PDF page \(index) could not be read")
        }
        let rect = page.getBoxRect(.cropBox)
        guard !rect.isEmpty, rect.width.isFinite, rect.height.isFinite else {
          throw ManagerError.core("Invalid PDF page geometry")
        }
        let rotated = abs(page.rotationAngle) % 180 == 90
        let width = rotated ? rect.height : rect.width
        let height = rotated ? rect.width : rect.height
        let scale = min(2, 2048 / max(width, height))
        let pixelsW = max(1, Int(ceil(width * scale)))
        let pixelsH = max(1, Int(ceil(height * scale)))
        guard
          let context = CGContext(
            data: nil, width: pixelsW, height: pixelsH, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
          )
        else {
          throw ManagerError.core("Not enough memory to render PDF page \(index)")
        }
        let destination = CGRect(x: 0, y: 0, width: pixelsW, height: pixelsH)
        context.setFillColor(CGColor(gray: 1, alpha: 1))
        context.fill(destination)
        context.concatenate(
          page.getDrawingTransform(
            .cropBox, rect: destination, rotate: 0, preserveAspectRatio: true))
        context.drawPDFPage(page)
        guard let image = context.makeImage() else {
          throw ManagerError.core("PDF rasterization failed")
        }
        let data = NSMutableData()
        guard
          let output = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil)
        else { throw ManagerError.core("PDF image encoding failed") }
        CGImageDestinationAddImage(
          output, image, [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary)
        guard CGImageDestinationFinalize(output) else {
          throw ManagerError.core("PDF image encoding failed")
        }
        return data as Data
      }
      result.append(
        .object([
          "type": .string("input_text"),
          "text": .string("[\(metadata["name"]?.string ?? "PDF"), page \(index)]"),
        ]))
      result.append(
        .object([
          "type": .string("input_image"), "detail": .string("auto"),
          "image_url": .string("data:image/jpeg;base64,\(encoded.base64EncodedString())"),
        ]))
    }
    return result
  }
}
