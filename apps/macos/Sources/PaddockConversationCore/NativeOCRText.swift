import Foundation

enum NativeOCRText {
  // Same closed-marker forms as web ocr.ts. Incomplete streaming markers
  // remain untouched; the saved raw extraction always stays authoritative.
  private static let forms: [(NSRegularExpression, String)] = [
    (#"<\|ref\|>([\s\S]*?)<\|/ref\|><\|det\|>[\s\S]*?<\|/det\|>"#, "$1"),
    (#"<\|det\|>\s*[A-Za-z_][\w-]*\s*\[[^\]]*?\]\s*<\|/det\|>\s*"#, ""),
    (#"(?:<\|LOC_(?:BEGIN|END|SEP|\d+)\|>)+"#, ""),
    (#"<\|grounding\|>"#, ""),
  ].map { (try! NSRegularExpression(pattern: $0.0), $0.1) }

  static func display(_ text: String) -> String {
    let cleaned =
      text.contains("<|")
      ? forms.reduce(text) { value, form in
        form.0.stringByReplacingMatches(
          in: value, range: NSRange(value.startIndex..., in: value), withTemplate: form.1)
      } : text
    return NativeOCRTables.display(cleaned)
  }
}
