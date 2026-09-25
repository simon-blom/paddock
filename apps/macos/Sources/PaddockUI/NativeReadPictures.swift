import AppKit
import ImageIO
import PaddockConversationCore
import SwiftUI
import UniformTypeIdentifiers

enum NativeReadPictures {
  enum Source: Sendable {
    case file(URL)
    case bytes(Data, name: String)

    nonisolated func prepare() throws -> ReadPicture {
      switch self {
      case .file(let url): try NativeReadPictures.prepare(url)
      case .bytes(let data, let name): try NativeReadPictures.prepare(data, name: name)
      }
    }
  }
  static let maximumSourceBytes = 32 * 1024 * 1024
  /// ImageIO downsamples while decoding, applies EXIF orientation, and strips
  /// metadata on recompression. No full-resolution bitmap or main-actor decode.
  nonisolated static func prepare(_ url: URL) throws -> ReadPicture {
    guard url.isFileURL else { throw ConversationFailure.invalid("Choose a local image.") }
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    let data = try handle.read(upToCount: maximumSourceBytes + 1) ?? Data()
    return try prepare(data, name: url.lastPathComponent)
  }

  nonisolated static func prepare(_ data: Data, name: String) throws -> ReadPicture {
    guard data.count <= maximumSourceBytes else { throw ConversationFailure.tooLarge }
    return try autoreleasepool {
      guard
        let source = CGImageSourceCreateWithData(
          data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
        let type = CGImageSourceGetType(source),
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
        let width = properties[kCGImagePropertyPixelWidth] as? Int,
        let height = properties[kCGImagePropertyPixelHeight] as? Int,
        (1...65536).contains(width), (1...65536).contains(height)
      else { throw ConversationFailure.invalid("This image could not be decoded.") }
      let mime = UTType(type as String)?.preferredMIMEType ?? ""
      if max(width, height) <= 2048, data.count <= 3 * 1024 * 1024,
        ["image/png", "image/jpeg", "image/webp"].contains(mime)
      {
        return ReadPicture(
          name: name, url: "data:\(mime);base64,\(data.base64EncodedString())")
      }
      guard
        let pixels = CGImageSourceCreateThumbnailAtIndex(
          source, 0,
          [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 2048,
            kCGImageSourceShouldCacheImmediately: true,
          ] as CFDictionary)
      else { throw ConversationFailure.invalid("This image could not be resized.") }
      let bytes = NSMutableData()
      guard
        let destination = CGImageDestinationCreateWithData(
          bytes, UTType.jpeg.identifier as CFString, 1, nil)
      else {
        throw ConversationFailure.invalid("The image copy could not be created.")
      }
      CGImageDestinationAddImage(
        destination, pixels, [kCGImageDestinationLossyCompressionQuality: 0.92] as CFDictionary)
      guard CGImageDestinationFinalize(destination), bytes.length <= 3 * 1024 * 1024 else {
        throw ConversationFailure.invalid(
          "The resized image is still too large. Choose a smaller image.")
      }
      return ReadPicture(
        name: name,
        url: "data:image/jpeg;base64,\((bytes as Data).base64EncodedString())")
    }
  }
}

struct NativeReadPictureChip: View {
  let picture: ReadPicture
  var remove: (() -> Void)?
  @State private var thumbnail: NSImage?
  var body: some View {
    HStack(spacing: 8) {
      Group {
        if let thumbnail {
          Image(nsImage: thumbnail).resizable().scaledToFit()
        } else {
          Image(systemName: "photo").foregroundStyle(.secondary)
        }
      }.frame(width: 44, height: 44)
      Text(picture.name).lineLimit(1).truncationMode(.middle).help(picture.name)
      if let remove {
        Button("Remove image", systemImage: "xmark", action: remove)
          .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).accessibilityLabel(
            "Remove \(picture.name)")
      }
    }.padding(8).background(
      PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
    )
    .task(id: picture.ref) {
      let url = picture.url
      let bytes = await Task.detached(priority: .utility) {
        guard let comma = url.firstIndex(of: ","),
          let data = Data(base64Encoded: String(url[url.index(after: comma)...])),
          let source = CGImageSourceCreateWithData(
            data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
          let image = CGImageSourceCreateThumbnailAtIndex(
            source, 0,
            [
              kCGImageSourceCreateThumbnailFromImageAlways: true,
              kCGImageSourceCreateThumbnailWithTransform: true,
              kCGImageSourceThumbnailMaxPixelSize: 96,
            ] as CFDictionary)
        else { return Data() }
        let bytes = NSMutableData()
        guard
          let dest = CGImageDestinationCreateWithData(
            bytes, UTType.png.identifier as CFString, 1, nil)
        else { return Data() }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest) ? bytes as Data : Data()
      }.value
      guard !Task.isCancelled else { return }
      thumbnail = NSImage(data: bytes)
    }
  }
}
