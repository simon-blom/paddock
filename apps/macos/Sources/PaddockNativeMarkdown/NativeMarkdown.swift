import AppKit
import MarkdownView
import PaddockDesign
import SwiftUI

/// One retained incremental parser per message. A completed source cannot be
/// reused for a continuation: replacing it then starts a fresh parsing cycle.
public struct NativeMarkdown: View, Equatable {
  public let text: String
  public let streaming: Bool
  public let textSize: CGFloat
  @State private var source = StreamingMarkdownSource()
  @State private var finished = false
  @State private var renderedHeight: CGFloat?

  public init(_ text: String, streaming: Bool = false, textSize: CGFloat = 15) {
    self.text = text
    self.streaming = streaming
    self.textSize = textSize
  }
  public nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
    lhs.text == rhs.text && lhs.streaming == rhs.streaming && lhs.textSize == rhs.textSize
  }
  public var body: some View {
    StreamingMarkdownReader(source) { result in
      // A single AppKit text surface, rather than independently selectable
      // SwiftUI blocks. Embedded code, math and diagrams keep their renderers.
      SelectableMarkdownSurface {
        MarkdownText(result)
          .markdownCodeBlockStyle(NativeCodeBlockStyle(streaming: streaming))
          .textSelection(.enabled)
          .fixedSize(horizontal: false, vertical: true)
          .onGeometryChange(for: CGFloat.self) {
            $0.size.height
          } action: { height in
            // Cross the nested AppKit hosting boundary explicitly. Otherwise
            // a lazy parent can retain the empty parse's height until an
            // unrelated composer/selection update, clipping the first answer.
            if height.isFinite && height >= 0 { renderedHeight = height }
          }
      }
      .frame(height: renderedHeight)
    }
    .markdownFontGroup(ChatMarkdownFonts(size: textSize))
    .font(.system(size: textSize))
    .foregroundStyle(PaddockAppearance.primary)
    .tint(PaddockAppearance.accent)
    .lineSpacing(4)
    .markdownComponentSpacing(12)
    .tint(.primary, for: .inlineCodeBlock)
    .markdownMathRenderingEnabled()
    .markdownStreamingRenderThrottle(.milliseconds(33))
    // Loading arbitrary assistant-authored image URLs is not part of this
    // renderer test. Original attachments keep their authenticated web viewer.
    .markdownElementRenderer(.image(DeferredImage(), urlScheme: "https"))
    .markdownElementRenderer(.image(DeferredImage(), urlScheme: "http"))
    .markdownElementRenderer(.image(DeferredImage(), urlScheme: "file"))
    .environment(
      \.openURL,
      OpenURLAction { url in
        guard ["https", "http", "mailto"].contains(url.scheme?.lowercased() ?? "") else {
          return .discarded
        }
        return .systemAction
      }
    )
    .task(id: RenderInput(text: text, streaming: streaming)) {
      let requested = text
      let value = await Task.detached(priority: .userInitiated) {
        NativeMarkdownPolicy.source(requested)
      }.value
      guard !Task.isCancelled else { return }
      update(value)
    }
  }
  private struct RenderInput: Equatable {
    let text: String
    let streaming: Bool
  }
  private func update(_ value: String) {
    if finished && (source.text != value || streaming) {
      source = StreamingMarkdownSource(value)
      finished = false
    } else {
      source.text = value
    }
    if !streaming {
      source.finishStreaming()
      finished = true
    }
  }
}

/// Do not inherit the library's smaller macOS body text or serif block quotes.
/// Platform fonts also keep the same metrics on our macOS 15 deployment target.
private struct ChatMarkdownFonts: MarkdownFontGroup {
  let size: CGFloat
  var body: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size) }
  var h1: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size + 9, weight: .semibold) }
  var h2: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size + 5, weight: .semibold) }
  var h3: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size + 2, weight: .semibold) }
  var h4: any CustomCTFontConvertible { NSFont.systemFont(ofSize: size, weight: .semibold) }
  var h5: any CustomCTFontConvertible { h4 }
  var h6: any CustomCTFontConvertible { h4 }
  var blockQuote: any CustomCTFontConvertible { body }
  var codeBlock: any CustomCTFontConvertible {
    NSFont.monospacedSystemFont(ofSize: size - 2, weight: .regular)
  }
  var tableBody: any CustomCTFontConvertible { body }
  var tableHeader: any CustomCTFontConvertible { h4 }
  var inlineMath: any CustomCTFontConvertible { body }
  var displayMath: any CustomCTFontConvertible { body }
}

private struct NativeCodeBlockStyle: MarkdownCodeBlockStyle {
  let streaming: Bool
  @ViewBuilder func makeBody(configuration: Configuration) -> some View {
    if configuration.language?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      == "mermaid"
    {
      NativeMermaid(source: configuration.code, streaming: streaming)
    } else {
      NativeCodeBlock(
        code: configuration.code, language: configuration.language, streaming: streaming)
    }
  }
}

private struct DeferredImage: MarkdownImageRenderer {
  func makeBody(configuration: Configuration) -> some View {
    Label(configuration.alternativeText ?? "Image", systemImage: "photo")
      .font(.callout).foregroundStyle(.secondary)
      .help(
        "Remote images are not fetched automatically. Open an attached original in the native image viewer."
      )
  }
}
