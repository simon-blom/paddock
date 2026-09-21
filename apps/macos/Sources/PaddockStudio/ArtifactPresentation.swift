import Foundation
import PaddockClient

public enum ArtifactPresentation {
  public struct Group: Identifiable {
    public let model: String
    public let items: [StudioState.Artifact]
    public var id: String { model }
  }
  public static func groups(_ items: [StudioState.Artifact], laneOrder: [String]) -> [Group] {
    var order: [String] = []
    for item in items where !item.model.isEmpty && !order.contains(item.model) {
      order.append(item.model)
    }
    order.sort {
      (laneOrder.firstIndex(of: $0) ?? Int.max) < (laneOrder.firstIndex(of: $1) ?? Int.max)
    }
    return order.map { model in Group(model: model, items: items.filter { $0.model == model }) }
  }
  public static func splits(groups: Int, width: Double) -> Bool {
    groups >= 2 && width.isFinite && Int(max(0, width) / 340) >= groups
  }
  public static func identity(_ artifact: StudioState.Artifact, state: StudioState?) -> (
    name: String, vendor: String
  ) {
    let model = state?.models.first { $0.id == artifact.model }
    let saved = state?.nativeTranscript?.messages.first {
      $0.role == "assistant" && $0.model == artifact.model
    }?.chrome
    return (
      model?.title ?? saved?.modelName ?? CloudModelIdentity.fallbackName(artifact.model),
      model?.vendor.nonempty ?? saved?.vendor.nonempty ?? CloudModelIdentity.vendor(
        CloudModelIdentity.bareModel(artifact.model)) ?? ""
    )
  }
  public static func language(_ artifact: StudioState.Artifact) -> String {
    if !artifact.language.isEmpty { return artifact.language }
    return [
      "html": "html", "svg": "xml", "markdown": "markdown", "mermaid": "mermaid", "graph": "cypher",
      "csv": "csv",
    ][artifact.kind] ?? ""
  }
  public static func filename(_ artifact: StudioState.Artifact) -> String {
    let ext =
      [
        "html": "html", "svg": "svg", "markdown": "md", "mermaid": "mmd", "csv": "csv",
        "graph": "cypher",
      ][artifact.kind] ?? artifact.language.nonempty ?? "txt"
    let safe = artifact.title.replacingOccurrences(
      of: #"[^\w.-]+"#, with: "-", options: .regularExpression
    ).trimmingCharacters(in: CharacterSet(charactersIn: ".-"))
    let suffix = ext.replacingOccurrences(
      of: #"[^a-zA-Z0-9]+"#, with: "", options: .regularExpression)
    return (safe.isEmpty ? artifact.id : safe) + "." + (suffix.isEmpty ? "txt" : suffix)
  }
  public static func fence(_ source: String, language: String) -> String {
    var longest = 0
    var run = 0
    for c in source {
      run = c == "`" ? run + 1 : 0
      longest = max(longest, run)
    }
    let delimiter = String(repeating: "`", count: max(3, longest + 1))
    return "\(delimiter)\(language)\n\(source)\n\(delimiter)"
  }
}

extension String { fileprivate var nonempty: String? { isEmpty ? nil : self } }

/// RFC4180-style reader, off-main. Retain only the preview's bounded cells;
/// scan the rest to report the exact hidden row/column counts. Original source
/// is untouched and remains available for editing/export.
public struct ArtifactCSV: Sendable {
  public let rows: [[String]]
  public let totalRows: Int
  public let columns: Int
  public let clippedCells: Bool
  public static func parse(_ source: String) throws -> Self {
    guard source.utf8.count <= 4 * 1024 * 1024 else { throw CSVError() }
    let bytes = Array(source.utf8)
    var rows: [[String]] = []
    var row: [String] = []
    var cell: [UInt8] = []
    var total = 0
    var columns = 0
    var column = 0
    var i = 0
    var quoted = false
    var cellNonempty = false
    var clippedCells = false
    func finishCell() {
      if total <= 500 && column < 100 {
        let text = String(decoding: cell, as: UTF8.self)
        if text.count > 4096 { clippedCells = true }
        row.append(String(text.prefix(4096)))
      }
      cell = []
      column += 1
    }
    func finishRow() {
      let nonempty = column > 0 || cellNonempty
      finishCell()
      if nonempty {
        columns = max(columns, column)
        if total <= 500 { rows.append(row) }
        total += 1
      }
      row = []
      column = 0
      cellNonempty = false
    }
    while i < bytes.count {
      if i % 4096 == 0 { try Task.checkCancellation() }
      let c = bytes[i]
      if c == 34 {
        if quoted && i + 1 < bytes.count && bytes[i + 1] == 34 {
          if total <= 500 && column < 100 { cell.append(34) }
          cellNonempty = true
          i += 1
        } else {
          quoted.toggle()
        }
      } else if c == 44 && !quoted {
        finishCell()
      } else if (c == 10 || c == 13) && !quoted {
        if c == 13 && i + 1 < bytes.count && bytes[i + 1] == 10 { i += 1 }
        finishRow()
      } else {
        cellNonempty = true
        if total <= 500 && column < 100 { cell.append(c) }
      }
      i += 1
    }
    if cellNonempty || column > 0 { finishRow() }
    return Self(rows: rows, totalRows: total, columns: columns, clippedCells: clippedCells)
  }
}

private struct CSVError: LocalizedError {
  var errorDescription: String? {
    "CSV preview is limited to 4 MiB. Source and export include the full file."
  }
}
