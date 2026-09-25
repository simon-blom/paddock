import AppKit
import PaddockConversationCore
import UniformTypeIdentifiers

/// Called only for an explicit paste into Reads' State editor. Snapshot the
/// selected representations before returning to the event loop; decoding and
/// resizing use the same off-main pipeline as files. Never read remote URLs.
@MainActor enum NativeReadClipboard {
  private enum Item {
    case file(URL)
    case image(NSPasteboardItem, NSPasteboard.PasteboardType)
  }

  private static func items(_ board: NSPasteboard) -> [Item] {
    (board.pasteboardItems ?? []).compactMap { item in
      // A Finder preview must not replace the original file, or turn a copied
      // PDF/folder into an image. Multiple representations are ONE attachment.
      if let file = item.string(forType: .fileURL) {
        guard let url = URL(string: file), url.isFileURL,
          UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) == true
        else { return nil }
        return .file(url)
      }
      let preferred: [NSPasteboard.PasteboardType] = [.png, .init(UTType.jpeg.identifier), .tiff]
      let type =
        preferred.first { item.types.contains($0) }
        ?? item.types.first { UTType($0.rawValue)?.conforms(to: .image) == true }
      return type.map { .image(item, $0) }
    }
  }

  static func containsImages(_ board: NSPasteboard) -> Bool { !items(board).isEmpty }

  static func snapshot(_ board: NSPasteboard, remaining: Int) throws -> [NativeReadPictures.Source]
  {
    let selected = items(board)
    guard selected.count <= remaining else {
      throw ConversationFailure.invalid("A read takes up to 16 images.")
    }
    var retained = 0
    return try selected.enumerated().map { index, item in
      switch item {
      case .file(let url): return .file(url)
      case .image(let item, let type):
        guard let bytes = item.data(forType: type) else {
          throw ConversationFailure.invalid("The clipboard image could not be read. Copy it again.")
        }
        retained += bytes.count
        guard retained <= NativeReadPictures.maximumSourceBytes else {
          throw ConversationFailure.invalid(
            "The clipboard images exceed 32 MiB. Paste fewer or smaller images.")
        }
        return .bytes(
          bytes, name: selected.count == 1 ? "Pasted image" : "Pasted image \(index + 1)")
      }
    }
  }
}
