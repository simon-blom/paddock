import AppKit
import PaddockClient
import PaddockConversationCore
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native Reads layout", .serialized) @MainActor
struct NativeReadsLayoutTests {
  @Test func nativeMenuCannotExpandProviderArtworkToItsOriginalSVGSize() {
    for vendor in ProviderArtwork.names.keys.sorted() {
      let host = NSHostingController(
        rootView: Menu {
          Button("Select") {}
        } label: {
          HStack(spacing: 8) {
            ModelProviderLogo(vendor: vendor)
            Text("DiffusionGemma 26B A4B")
          }
        }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize())
      let size = host.sizeThatFits(in: NSSize(width: 600, height: 600))
      #expect(size.height <= 36, "Oversized native menu trigger for \(vendor): \(size)")
      #expect(size.width <= 340)
    }
  }

  @Test func readsFormFitsNarrowAndWideWindowsInBothThemes() async throws {
    _ = NSApplication.shared
    let model = NativeReadsModel(client: NativeManager())
    model.api = { path, _, _, _ in
      let raw: String
      if path == "api/runners" {
        raw =
          #"[{"port":1234,"model":"diffusiongemma","display":"DiffusionGemma 26B A4B","vendor":"Google"}]"#
      } else if path.hasSuffix("/server") {
        raw =
          #"{"structured_read":{"canvas_width":256,"max_questions":64,"max_samples":32,"types":["noul","choice","score"]}}"#
      } else {
        raw = "[]"
      }
      return try JSONDecoder().decode(ConversationValue.self, from: Data(raw.utf8))
    }
    await model.refresh()
    for dark in [false, true] {
      for width: CGFloat in [440, 680, 1200] {
        let host = NSHostingController(
          rootView: NativeReadsView(model: model, onStart: {})
            .environment(\.colorScheme, dark ? .dark : .light))
        host.sizingOptions = []
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: width, height: 1000),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentViewController = host
        window.setContentSize(NSSize(width: width, height: 1000))
        window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(100))
        host.view.layoutSubtreeIfNeeded()
        #expect(abs(host.view.frame.width - width) < 1)
        let scroll = try #require(allViews(host.view).compactMap { $0 as? NSScrollView }.first)
        #expect((scroll.documentView?.frame.width ?? 0) <= width + 1)
        for control in allViews(host.view).compactMap({ $0 as? NSPopUpButton }) {
          #expect(control.frame.height <= 36)
          if let image = control.image { #expect(image.size.height <= 18) }
        }
        if let folder = ProcessInfo.processInfo.environment["PADDOCK_READS_SNAPSHOTS"],
          let bitmap = host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds)
        {
          host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
          try bitmap.representation(using: .png, properties: [:])?.write(
            to: URL(fileURLWithPath: folder).appending(
              path: "reads-\(Int(width))-\(dark ? "dark" : "light").png"))
        }
      }
    }
  }

  @Test func everyQuestionTypeFitsTheNarrowColumn() {
    for kind in ReadQuestion.Kind.allCases {
      let question = ReadQuestion(questionID: "question_id", kind: kind)
      let host = NSHostingController(
        rootView: NativeReadQuestionRow(
          question: .constant(question), onDuplicate: {}, onMove: { _ in }, onRemove: {}))
      let size = host.sizeThatFits(in: NSSize(width: 356, height: 1000))
      #expect(size.width <= 356, "Question controls must fit the column: \(kind), \(size)")
      #expect(size.height < 380)
    }
  }

  private func allViews(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(allViews)
  }
}
