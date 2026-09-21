import AppKit
import ImageIO
import PaddockStudio
import SwiftUI

struct NativeImagePreview: View {
  let document: StudioState.Document
  let workspace: StudioWorkspace
  @State private var image: CGImage?
  @State private var details = ""
  @State private var error: String?
  @State private var zoom: CGFloat = 1
  @State private var showDetails = false
  var body: some View {
    VStack(spacing: 0) {
      HStack {
        Text(document.name).lineLimit(1).truncationMode(.middle)
        Spacer()
        Button("Zoom out", systemImage: "minus.magnifyingglass") { zoom = max(0.25, zoom / 1.25) }
        Button("Zoom in", systemImage: "plus.magnifyingglass") { zoom = min(8, zoom * 1.25) }
        Button("Fit") { zoom = 1 }
        Button("Image details", systemImage: "info.circle") { showDetails.toggle() }
        Button("Save original", systemImage: "square.and.arrow.down") {
          Task { await workspace.saveOriginal(document.id, name: document.name) }
        }
        Button("Close image", systemImage: "xmark") {
          Task { await workspace.perform("closePreview") }
        }
      }.font(.system(size: 12)).labelStyle(.iconOnly).buttonStyle(.plain).padding(12)
      if showDetails {
        PaddockScrollView {
          Text(verbatim: details).font(.system(size: 12, design: .monospaced)).textSelection(
            .enabled
          ).padding()
        }
      } else if let image {
        GeometryReader { geometry in
          PaddockScrollView([.horizontal, .vertical]) {
            Image(decorative: image, scale: 1).resizable().scaledToFit()
              .frame(
                width: max(1, geometry.size.width - 24) * zoom,
                height: max(1, geometry.size.height - 24) * zoom
              ).padding(12)
              .accessibilityLabel(document.name)
          }
        }
      } else if let error {
        ContentUnavailableView("Image unavailable", systemImage: "photo", description: Text(error))
      } else {
        ProgressView("Opening image…").frame(maxWidth: .infinity, maxHeight: .infinity)
      }
    }.background(PaddockStyle.canvas).accessibilityIdentifier("native-image-preview")
      .task(id: document.id) {
        image = nil
        error = nil
        zoom = 1
        do {
          let file = try await workspace.downloadOriginal(document.id)
          defer { try? FileManager.default.removeItem(at: file) }
          let decoded = try await Task.detached {
            guard let source = CGImageSourceCreateWithURL(file as CFURL, nil),
              let pixels = CGImageSourceCreateThumbnailAtIndex(
                source, 0,
                [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,
                  kCGImageSourceThumbnailMaxPixelSize: 4096,
                  kCGImageSourceShouldCacheImmediately: true,
                ] as CFDictionary)
            else { throw ImageError.decode }
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil)
            return DecodedImage(
              image: pixels, details: properties.map { String(describing: $0) } ?? "No metadata")
          }.value
          try Task.checkCancellation()
          image = decoded.image
          details = decoded.details
        } catch is CancellationError {} catch { self.error = error.localizedDescription }
      }
  }
  private enum ImageError: LocalizedError {
    case decode
    var errorDescription: String? {
      "macOS could not decode this image. The saved original is unchanged."
    }
  }
  private struct DecodedImage: @unchecked Sendable {
    let image: CGImage
    let details: String
  }
}
