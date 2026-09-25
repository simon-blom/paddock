import AppKit
import ImageIO
import PaddockConversationCore
import SwiftUI
import Testing
import UniformTypeIdentifiers

@testable import PaddockUI

@Suite("Reads clipboard", .serialized) @MainActor
struct NativeReadClipboardTests {
  private func board() -> NSPasteboard { NSPasteboard.withUniqueName() }
  private func image(_ type: UTType = .png, width: Int = 32, height: Int = 24) throws -> Data {
    let context = try #require(
      CGContext(
        data: nil, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
    context.setFillColor(CGColor(red: 0.2, green: 0.5, blue: 0.9, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let pixels = try #require(context.makeImage())
    let bytes = NSMutableData()
    let destination = try #require(
      CGImageDestinationCreateWithData(bytes, type.identifier as CFString, 1, nil))
    CGImageDestinationAddImage(destination, pixels, nil)
    #expect(CGImageDestinationFinalize(destination))
    return bytes as Data
  }
  private func model(vision: Bool = true) async -> NativeReadsModel {
    let model = NativeReadsTests().model()
    let original = model.api
    model.api = { path, method, body, query in
      if path.hasSuffix("1234/server") {
        return .object([
          "structured_read": .object(["canvas_width": .number(256), "images": .bool(vision)])
        ])
      }
      return try await original(path, method, body, query)
    }
    await model.refresh()
    return model
  }

  @Test func screenshotRepresentationsAttachOnceAndSnapshotBeforeClipboardChanges() async throws {
    let board = board()
    defer { board.releaseGlobally() }
    let png = try image()
    board.setData(png, forType: .png)
    board.setData(try image(.tiff), forType: .tiff)
    board.setString("Accompanying alt text", forType: .string)
    let model = await model()
    model.draft.state = "Keep my instructions"
    #expect(model.pastePictures(board) && model.importing)
    board.clearContents()
    board.setString("The clipboard changed", forType: .string)
    await model.settle()
    let attached = try #require(model.draft.images.first)
    #expect(model.draft.images.count == 1 && model.stateError == nil && !model.importing)
    #expect(attached.url == "data:image/png;base64,\(png.base64EncodedString())")
    #expect(model.draft.state == "Keep my instructions" && model.canRun)
    #expect(model.draft.request(model: "diffusion")["images"] == .array([.string(attached.url)]))
    #expect(
      try ReadPicture.restore(
        .object(["images": .array([attached.historyReference])]),
        table: .object([attached.ref: .string(attached.url)])) == [attached])
  }

  @Test func tiffScreenshotAndJpegAreDecodedOffMainThroughTheSamePipeline() async throws {
    for type in [UTType.tiff, .jpeg] {
      let board = board()
      defer { board.releaseGlobally() }
      board.setData(try image(type, width: 4096, height: 64), forType: .init(type.identifier))
      let model = await model()
      #expect(model.pastePictures(board))
      await model.settle()
      let url = try #require(model.draft.images.first?.url)
      let data = try #require(Data(base64Encoded: String(url.split(separator: ",")[1])))
      let source = try #require(CGImageSourceCreateWithData(data as CFData, nil))
      let props = try #require(
        CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
      #expect(props[kCGImagePropertyPixelWidth] as? Int == 2048)
      #expect(props[kCGImagePropertyPixelHeight] as? Int == 32)
      #expect(model.stateError == nil)
    }
  }

  @Test func finderOriginalsWinOverPreviewAndNonImageFilesAreNotAttached() throws {
    let board = board()
    defer { board.releaseGlobally() }
    let file = NSPasteboardItem()
    file.setString(URL(fileURLWithPath: "/tmp/original.heic").absoluteString, forType: .fileURL)
    file.setData(try image(), forType: .png)
    let pdf = NSPasteboardItem()
    pdf.setString(URL(fileURLWithPath: "/tmp/document.pdf").absoluteString, forType: .fileURL)
    pdf.setData(try image(), forType: .png)
    board.writeObjects([file, pdf])
    let sources = try NativeReadClipboard.snapshot(board, remaining: 16)
    #expect(sources.count == 1)
    guard case .file(let url) = sources.first else {
      Issue.record("Original file was replaced by preview")
      return
    }
    #expect(url.lastPathComponent == "original.heic")
  }

  @Test func ordinaryTextAndRemoteURLsAreNotIntercepted() async {
    let model = await model()
    let board = board()
    defer { board.releaseGlobally() }
    board.setString("Ordinary text", forType: .string)
    #expect(!model.pastePictures(board) && !model.importing)
    board.clearContents()
    board.setString("https://example.com/image.png", forType: .URL)
    #expect(!model.pastePictures(board) && model.draft.images.isEmpty)
  }

  @Test func multipleImagesStayOrderedAndCapacityFailureIsAtomic() async throws {
    let board = board()
    defer { board.releaseGlobally() }
    let originals = try [image(width: 20), image(width: 21)]
    let items = originals.map { data in
      let item = NSPasteboardItem()
      item.setData(data, forType: .png)
      return item
    }
    board.writeObjects(items)
    let model = await model()
    #expect(model.pastePictures(board))
    // A repeated command cannot append a second batch while the first awaits decoding.
    #expect(model.pastePictures(board))
    await model.settle()
    #expect(
      model.draft.images.map(\.url)
        == originals.map { "data:image/png;base64,\($0.base64EncodedString())" })
    model.draft.images = Array(repeating: model.draft.images[0], count: 15)
    let before = model.draft
    #expect(model.pastePictures(board))
    await model.settle()
    #expect(model.draft == before && model.stateError != nil && !model.importing)
  }

  @Test func corruptImagesAndUnsupportedModelsNeverSilentlyInsertPaths() async throws {
    let board = board()
    defer { board.releaseGlobally() }
    board.setData(Data("not an image".utf8), forType: .png)
    let capable = await model()
    #expect(capable.pastePictures(board))
    await capable.settle()
    #expect(capable.draft.images.isEmpty && capable.stateError != nil && !capable.importing)
    let textOnly = await model(vision: false)
    #expect(textOnly.pastePictures(board))
    #expect(textOnly.draft.images.isEmpty && textOnly.stateError != nil && !textOnly.importing)
    board.clearContents()
    board.setData(Data(count: NativeReadPictures.maximumSourceBytes + 1), forType: .tiff)
    #expect(throws: (any Error).self) { try NativeReadClipboard.snapshot(board, remaining: 16) }
  }

  @Test func stateEditorKeepsSelectionTextPasteUndoAndImageOnlyPasteEligibility() async throws {
    _ = NSApplication.shared
    var text = "Keep these words"
    let model = await model()
    let host = NSHostingController(
      rootView: NativeReadStateEditor(
        text: Binding(get: { text }, set: { text = $0 }), onPasteImages: model.pastePictures))
    let window = NSWindow(
      contentRect: NSRect(x: -12000, y: -12000, width: 400, height: 160),
      styleMask: [.titled], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentViewController = host
    defer { window.close() }
    host.view.frame = NSRect(x: 0, y: 0, width: 400, height: 160)
    host.view.layoutSubtreeIfNeeded()
    func find(_ view: NSView) -> ReadStateTextView? {
      (view as? ReadStateTextView) ?? view.subviews.lazy.compactMap(find).first
    }
    let editor = try #require(find(host.view))
    #expect(editor.textLayoutManager != nil && editor.readablePasteboardTypes.contains(.png))
    window.makeFirstResponder(editor)
    let selection = NSRange(location: 5, length: 5)
    editor.setSelectedRange(selection)
    let board = board()
    defer { board.releaseGlobally() }
    board.setData(try image(), forType: .png)
    #expect(editor.acceptImages(board))
    await model.settle()
    #expect(
      text == "Keep these words" && editor.string == text && editor.selectedRange() == selection)
    board.clearContents()
    board.setString("edited", forType: .string)
    #expect(!editor.acceptImages(board))
    let undo = try #require(editor.undoManager)
    undo.beginUndoGrouping()
    #expect(editor.readSelection(from: board))
    undo.endUndoGrouping()
    #expect(text == "Keep edited words")
    undo.undo()
    #expect(text == "Keep these words")
  }
}
