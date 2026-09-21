import AppKit
import PaddockDesign
import SwiftUI

/// Native lexical decoration, off-main and serialized. No highlight.js or JS
/// context. Unsupported languages retain exact source with plain styling.
actor NativeCodeHighlighter {
  static let shared = NativeCodeHighlighter()
  func highlight(_ code: String, language: String?, dark: Bool) throws -> AttributedString {
    try Task.checkCancellation()
    guard code.utf8.count <= 256 * 1024 else { return AttributedString(code) }
    let language = (language ?? "").lowercased()
    let hashComments = [
      "python", "py", "ruby", "rb", "shell", "sh", "bash", "zsh", "yaml", "yml", "toml",
    ].contains(language)
    let languages = [
      "swift", "rust", "rs", "javascript", "js", "typescript", "ts", "tsx", "jsx", "c", "cpp",
      "c++", "csharp", "cs", "java", "kotlin", "go", "json", "sql", "python", "py", "ruby", "rb",
      "shell", "sh", "bash", "zsh", "yaml", "yml", "toml",
    ]
    guard languages.contains(language) else { return AttributedString(code) }
    let source = code as NSString
    let styled = NSMutableAttributedString(string: code)
    // Token alternatives are ordered; matches are non-overlapping. Comments
    // inside strings and quoted keyword text are never independently recolored.
    let comment =
      language == "sql"
      ? "--[^\\r\\n]*" : hashComments ? "#[^\\r\\n]*" : "//[^\\r\\n]*|/\\*[\\s\\S]*?\\*/"
    let pattern =
      #"("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|("# + comment
      + #")|\b([0-9]+(?:\.[0-9]+)?)\b|\b([A-Za-z_][A-Za-z_0-9]*)\b"#
    let regex = try NSRegularExpression(pattern: pattern)
    let words = Set(
      "actor async await as associatedtype break case catch class const continue crate def defer delete do dyn else enum except export extends extension false final fn for from func function guard if impl import in init interface internal is let match module mut namespace new nil none null open override package private protocol pub public raise repeat return self static struct super switch throw throws trait true try type typealias typeof undefined union unsafe use using var virtual void when where while with yield"
        .split(separator: " ").map(String.init))
    let range = NSRange(location: 0, length: source.length)
    for (i, match) in regex.matches(in: code, range: range).enumerated() {
      if i % 256 == 0 { try Task.checkCancellation() }
      let color: NSColor?
      if match.range(at: 1).location != NSNotFound {
        color = dark ? .systemOrange : .systemRed
      } else if match.range(at: 2).location != NSNotFound {
        color = .secondaryLabelColor
      } else if match.range(at: 3).location != NSNotFound {
        color = .systemPurple
      } else if words.contains(source.substring(with: match.range).lowercased()) {
        color = .systemPurple
      } else {
        color = nil
      }
      if let color { styled.addAttribute(.foregroundColor, value: color, range: match.range) }
    }
    return AttributedString(styled)
  }
}

public struct NativeCodeBlock: View {
  let code: String
  let language: String?
  let streaming: Bool
  public init(code: String, language: String?, streaming: Bool) {
    self.code = code
    self.language = language
    self.streaming = streaming
  }
  @Environment(\.colorScheme) private var colorScheme
  @State private var highlighted: AttributedString?
  @State private var rendered: Input?
  @State private var copied = false
  private struct Input: Equatable {
    let code: String
    let language: String?
    let dark: Bool
  }
  private var input: Input { Input(code: code, language: language, dark: colorScheme == .dark) }
  public var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 12) {
        Text(language?.isEmpty == false ? language! : "Code")
        Spacer()
        Button(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "square.on.square") {
          NSPasteboard.general.clearContents()
          NSPasteboard.general.setString(code, forType: .string)
          copied = true
        }.buttonStyle(.plain).help("Copy code")
      }.font(.system(size: 11, weight: .medium)).foregroundStyle(PaddockAppearance.secondary)
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(PaddockAppearance.elevated)
      PaddockAppearance.border.frame(height: 1)
      PaddockScrollView(.horizontal) {
        NativeSelectableText(
          code, attributed: rendered == input ? highlighted : nil, size: 13, monospaced: true,
          wraps: false
        )
        .fixedSize(horizontal: true, vertical: true).padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
      }
    }.background(PaddockAppearance.surface)
      .clipShape(RoundedRectangle(cornerRadius: PaddockAppearance.Radius.card))
      .overlay(
        RoundedRectangle(cornerRadius: PaddockAppearance.Radius.card).strokeBorder(
          PaddockAppearance.border)
      )
      .task(id: input) {
        let requested = input
        do {
          if streaming { try await Task.sleep(for: .milliseconds(80)) }
          let value = try await NativeCodeHighlighter.shared.highlight(
            requested.code, language: requested.language, dark: requested.dark)
          try Task.checkCancellation()
          highlighted = value
          rendered = requested
        } catch is CancellationError {
          // A superseded highlight never replaces the current source.
        } catch {
          highlighted = nil
        }
      }
      .onChange(of: code) { _, _ in copied = false }
      .task(id: copied) {
        guard copied else { return }
        do {
          try await Task.sleep(for: .seconds(2))
          copied = false
        } catch {}
      }
  }
}
