import CoreGraphics
import Foundation
import ImageIO
import PaddockConversationCore
import Testing
import UniformTypeIdentifiers

@testable import PaddockUI

@Suite("Reads image preparation")
struct NativeReadPicturesTests {
  @Test func largeImageIsResizedWithoutModifyingOriginal() async throws {
    try await Task.detached {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
      try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
      defer { try? FileManager.default.removeItem(at: root) }
      let url = root.appendingPathComponent("large.png")
      let context = try #require(
        CGContext(
          data: nil, width: 4096, height: 64, bitsPerComponent: 8,
          bytesPerRow: 4096 * 4, space: CGColorSpaceCreateDeviceRGB(),
          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
      context.setFillColor(CGColor(red: 0.2, green: 0.5, blue: 0.9, alpha: 1))
      context.fill(CGRect(x: 0, y: 0, width: 4096, height: 64))
      let image = try #require(context.makeImage())
      let target = try #require(
        CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
      CGImageDestinationAddImage(target, image, nil)
      #expect(CGImageDestinationFinalize(target))
      let original = try Data(contentsOf: url)
      let picture = try NativeReadPictures.prepare(url)
      #expect(picture.name == "large.png" && picture.url.hasPrefix("data:image/jpeg;base64,"))
      let bytes = try #require(Data(base64Encoded: String(picture.url.split(separator: ",")[1])))
      let source = try #require(CGImageSourceCreateWithData(bytes as CFData, nil))
      let properties = try #require(
        CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
      #expect(properties[kCGImagePropertyPixelWidth] as? Int == 2048)
      #expect(properties[kCGImagePropertyPixelHeight] as? Int == 32)
      #expect(try Data(contentsOf: url) == original)
      #expect(picture.ref == ReadPicture.reference(picture.url))
      let invalid = root.appendingPathComponent("invalid.png")
      try Data("not pixels".utf8).write(to: invalid)
      #expect(throws: (any Error).self) { try NativeReadPictures.prepare(invalid) }
    }.value
  }
}
