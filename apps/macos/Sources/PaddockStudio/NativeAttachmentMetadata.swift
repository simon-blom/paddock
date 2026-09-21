import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum NativeAttachmentMetadata {
  /// Metadata/thumbnail decoding is off-main and does not instantiate a viewer.
  static func describe(_ url: URL, mime: String) async -> [String: StudioValue] {
    await Task.detached {
      var fields: [String: StudioValue] = [:]
      if mime == "application/pdf", let doc = CGPDFDocument(url as CFURL) {
        fields["pages"] = .number(Double(doc.numberOfPages))
      }
      if mime.hasPrefix("image/"),
        let source = CGImageSourceCreateWithURL(
          url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary)
      {
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        if let width = properties?[kCGImagePropertyPixelWidth] as? NSNumber {
          fields["width"] = .number(width.doubleValue)
        }
        if let height = properties?[kCGImagePropertyPixelHeight] as? NSNumber {
          fields["height"] = .number(height.doubleValue)
        }
        if mime == "image/tiff" { fields["pages"] = .number(Double(CGImageSourceGetCount(source))) }
        let options: [CFString: Any] = [
          kCGImageSourceCreateThumbnailFromImageAlways: true,
          kCGImageSourceThumbnailMaxPixelSize: 160,
          kCGImageSourceCreateThumbnailWithTransform: true,
        ]
        if let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) {
          let data = NSMutableData()
          if let destination = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil)
          {
            CGImageDestinationAddImage(
              destination, image, [kCGImageDestinationLossyCompressionQuality: 0.7] as CFDictionary)
            if CGImageDestinationFinalize(destination), data.length < 48 * 1024 {
              fields["thumbUrl"] = .string(
                "data:image/jpeg;base64,\((data as Data).base64EncodedString())")
            }
          }
        }
      }
      if mime.hasPrefix("audio/") || url.pathExtension.lowercased() == "webm" {
        let asset = AVURLAsset(url: url)
        if let duration = try? await asset.load(.duration), duration.seconds.isFinite,
          duration.seconds >= 0
        {
          fields["durationS"] = .number(duration.seconds)
        }
      }
      return fields
    }.value
  }
}
