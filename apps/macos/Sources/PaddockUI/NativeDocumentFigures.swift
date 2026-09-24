import AppKit
import PaddockStudio
import SwiftUI

enum NativeDocumentCrop {
  static func rectangle(_ box: [Double], width: Int, height: Int) -> CGRect? {
    guard box.count == 4, box.allSatisfy({ $0.isFinite && (0...999).contains($0) }),
      box[2] > box[0], box[3] > box[1], width > 0, height > 0
    else { return nil }
    return CGRect(
      x: box[0] / 999 * Double(width), y: box[1] / 999 * Double(height),
      width: (box[2] - box[0]) / 999 * Double(width),
      height: (box[3] - box[1]) / 999 * Double(height)
    ).integral.intersection(CGRect(x: 0, y: 0, width: width, height: height))
  }
}

/// Source pixels only: never an assistant-authored URL, never a web view.
/// All crops share the page buffer. Offscreen sections release their images;
/// the app-lifetime media service separately enforces its small reuse budget.
struct NativeDocumentFigures: View {
  typealias Page = StudioState.NativeTranscript.Message.DocumentResult.Page
  let page: Page
  let load: @MainActor () async throws -> CGImage
  let onOpen: () -> Void
  @State private var image: CGImage?
  @State private var error: String?
  @State private var visible = false
  @State private var showRegions = false
  @State private var retry = 0
  private var figures: [(String, [Double])] {
    Array(
      page.regions.lazy.filter { ["image", "figure", "picture"].contains($0.label.lowercased()) }
        .flatMap { region in region.boxes.lazy.map { (region.label, $0) } }
        .filter { NativeDocumentCrop.rectangle($0.1, width: 999, height: 999) != nil }
        .prefix(65))
  }
  private struct Request: Equatable {
    let visible: Bool
    let attachment: String
    let page: Int?
    let retry: Int
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 10) {
      if let image {
        ForEach(Array(figures.prefix(64).enumerated()), id: \.offset) { _, figure in
          if let rect = NativeDocumentCrop.rectangle(
            figure.1, width: image.width, height: image.height),
            let crop = image.cropping(to: rect)
          {
            Button(action: onOpen) {
              VStack(alignment: .leading, spacing: 4) {
                Image(decorative: crop, scale: 1).resizable().scaledToFit()
                  .frame(maxWidth: 600, maxHeight: 320, alignment: .leading)
                  .clipShape(RoundedRectangle(cornerRadius: 6))
                Text(figure.0.capitalized).font(.caption).foregroundStyle(.secondary)
              }
            }.buttonStyle(.plain).accessibilityLabel("Open \(figure.0) in original document")
          }
        }
        if figures.count > 64 {
          Button("More figures in original", action: onOpen).buttonStyle(
            .plain)
        }
      } else if !figures.isEmpty && error == nil {
        ProgressView().controlSize(.small).frame(height: 80)
      }
      DisclosureGroup(
        "\(page.regions.reduce(0) { $0 + $1.boxes.count }) mapped regions", isExpanded: $showRegions
      ) {
        if let image {
          Image(decorative: image, scale: 1).resizable().scaledToFit()
            .overlay {
              GeometryReader { geometry in
                Canvas { context, size in
                  var drawn = 0
                  regions: for region in page.regions {
                    for box in region.boxes {
                      if drawn == 1024 { break regions }
                      guard let unit = NativeDocumentCrop.rectangle(box, width: 999, height: 999)
                      else { continue }
                      let rect = CGRect(
                        x: unit.minX / 999 * size.width, y: unit.minY / 999 * size.height,
                        width: unit.width / 999 * size.width,
                        height: unit.height / 999 * size.height)
                      context.stroke(Path(rect), with: .color(.black.opacity(0.85)), lineWidth: 3)
                      context.stroke(Path(rect), with: .color(.white.opacity(0.95)), lineWidth: 1)
                      drawn += 1
                    }
                  }
                }.frame(width: geometry.size.width, height: geometry.size.height)
              }.allowsHitTesting(false)
            }
            .accessibilityLabel("Original page with \(page.regions.count) mapped regions")
        } else if error == nil {
          ProgressView().controlSize(.small).frame(height: 80)
        }
        Button("Open original", action: onOpen).buttonStyle(.plain).font(.caption)
        if page.regions.reduce(0, { $0 + $1.boxes.count }) > 1024 {
          Text("Preview shows the first 1,024 regions").font(.caption).foregroundStyle(.secondary)
        }
      }.font(.caption)
      if let error {
        HStack {
          Text(error).font(.caption).foregroundStyle(.secondary)
          Button("Retry") { retry += 1 }.buttonStyle(.plain)
        }
      }
    }
    .onScrollVisibilityChange(threshold: 0.01) { visible = $0 }
    .onDisappear {
      visible = false
      image = nil
    }
    .task(
      id: Request(
        visible: visible && (!figures.isEmpty || showRegions), attachment: page.attachmentID ?? "",
        page: page.pdfPage, retry: retry)
    ) {
      guard visible, !figures.isEmpty || showRegions, page.attachmentID != nil else {
        image = nil
        return
      }
      error = nil
      image = nil
      do {
        let decoded = try await load()
        try Task.checkCancellation()
        image = decoded
      } catch is CancellationError {} catch { self.error = error.localizedDescription }
    }
  }
}
