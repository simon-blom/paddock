import AppKit
import PaddockStudio
import SwiftUI

struct NativeCSVArtifact: View {
  let source: String
  @State private var table: ArtifactCSV?
  @State private var error: String?
  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if let table {
        ArtifactCSVTable(table: table)
        if table.totalRows > 501 || table.columns > 100 {
          Text(
            "Preview: up to 500 rows and 100 columns. Full file: \(max(0, table.totalRows - 1)) rows, \(table.columns) columns. Source and export include everything."
          )
          .font(.caption).foregroundStyle(.secondary)
        }
        if table.clippedCells {
          Text(
            "Long cells are shortened to 4,096 characters in the preview. Source and export include everything."
          )
          .font(.caption).foregroundStyle(.secondary)
        }
      } else if let error {
        Text(error).foregroundStyle(.secondary)
      } else {
        ProgressView()
      }
    }.frame(maxWidth: .infinity, maxHeight: .infinity)
      .task(id: source) {
        table = nil
        error = nil
        let task = Task.detached(priority: .userInitiated) { try ArtifactCSV.parse(source) }
        do {
          let result = try await withTaskCancellationHandler(
            operation: { try await task.value }, onCancel: { task.cancel() })
          try Task.checkCancellation()
          table = result
        } catch is CancellationError {} catch { self.error = error.localizedDescription }
      }
  }
}

/// AppKit reuses visible cells instead of mounting 50,000 SwiftUI text views.
private struct ArtifactCSVTable: NSViewRepresentable {
  let table: ArtifactCSV
  func makeCoordinator() -> Coordinator { Coordinator() }
  func makeNSView(context: Context) -> NSScrollView {
    let scroll = NSScrollView()
    let view = NSTableView()
    view.delegate = context.coordinator
    view.dataSource = context.coordinator
    view.rowHeight = 25
    view.usesAlternatingRowBackgroundColors = true
    view.columnAutoresizingStyle = .noColumnAutoresizing
    scroll.documentView = view
    scroll.hasVerticalScroller = true
    scroll.hasHorizontalScroller = true
    scroll.autohidesScrollers = true
    PaddockScrollbars.install(on: scroll)
    return scroll
  }
  func updateNSView(_ scroll: NSScrollView, context: Context) {
    guard let view = scroll.documentView as? NSTableView else { return }
    context.coordinator.rows = Array(table.rows.dropFirst())
    let count = min(100, table.columns)
    if view.tableColumns.count != count {
      view.tableColumns.forEach(view.removeTableColumn)
      for i in 0..<count {
        let column = NSTableColumn(identifier: .init(String(i)))
        column.width = 170
        column.minWidth = 60
        column.maxWidth = 800
        view.addTableColumn(column)
      }
    }
    for (i, column) in view.tableColumns.enumerated() {
      column.title =
        table.rows.first.flatMap { i < $0.count ? String($0[i].prefix(512)) : nil }
        ?? "Column \(i + 1)"
    }
    view.reloadData()
  }
  final class Coordinator: NSObject, NSTableViewDataSource, NSTableViewDelegate {
    var rows: [[String]] = []
    func numberOfRows(in tableView: NSTableView) -> Int { rows.count }
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int)
      -> NSView?
    {
      guard let id = tableColumn?.identifier, let column = Int(id.rawValue) else { return nil }
      let field =
        tableView.makeView(withIdentifier: id, owner: nil) as? NSTextField
        ?? NSTextField(labelWithString: "")
      field.identifier = id
      field.isSelectable = true
      field.lineBreakMode = .byTruncatingTail
      field.font = .systemFont(ofSize: 12)
      let value = column < rows[row].count ? rows[row][column] : ""
      field.stringValue = String(value.prefix(4096))
      field.toolTip = String(value.prefix(4096))
      return field
    }
  }
}
