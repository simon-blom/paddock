import Foundation
import Markdown

/// Plain, bounded notification text. Uses the renderer's Markdown parser;
/// never loads links/images, exposes URL targets, or renders HTML/code blocks.
public enum NotificationExcerpt {
  public static let characterLimit = 160
  public static func text(_ source: String) -> String? {
    guard !source.isEmpty else { return nil }
    let input = String(decoding: source.utf8.prefix(16 * 1024), as: UTF8.self)
    var walker = PlainText()
    walker.visit(Document(parsing: input))
    let plain = walker.result.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    guard !plain.isEmpty else { return nil }
    var result = ""
    var count = 0
    var bytes = 0
    for character in plain {
      let size = String(character).utf8.count
      guard count < characterLimit - 1, bytes + size <= 1000 else {
        return result.isEmpty ? nil : result.trimmingCharacters(in: .whitespaces) + "…"
      }
      result.append(character)
      count += 1
      bytes += size
    }
    return result
  }

  private struct PlainText: MarkupWalker {
    var result = ""
    mutating func visitText(_ text: Markdown.Text) { result += text.string }
    mutating func visitInlineCode(_ code: InlineCode) { result += code.code }
    mutating func visitSoftBreak(_ node: SoftBreak) { result += " " }
    mutating func visitLineBreak(_ node: LineBreak) { result += " " }
    mutating func visitParagraph(_ node: Paragraph) {
      descendInto(node)
      result += " "
    }
    mutating func visitHeading(_ node: Heading) {
      descendInto(node)
      result += " "
    }
    mutating func visitCodeBlock(_ node: CodeBlock) {}
    mutating func visitHTMLBlock(_ node: HTMLBlock) {}
    mutating func visitInlineHTML(_ node: InlineHTML) {}
    mutating func visitImage(_ node: Markdown.Image) {}
  }
}
