import PaddockClient
import SwiftUI

struct ConnectionsView: View {
  @Bindable var model: ConnectionsModel
  @State private var selection: String?
  @State private var search = ""
  @State private var removing: CloudConnection?
  var body: some View {
    let rows = model.rows.filter {
      search.isEmpty || $0.name.localizedCaseInsensitiveContains(search)
        || $0.baseUrl.localizedCaseInsensitiveContains(search)
    }
    let active = rows.first { $0.id == selection } ?? rows.first
    HSplitView {
      VStack(alignment: .leading, spacing: 14) {
        HStack {
          Text("Connections").font(.system(size: 17, weight: .semibold))
          Spacer()
          Button("Add endpoint", systemImage: "plus") { model.review() }
            .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).disabled(model.busy)
            .accessibilityIdentifier("connection-add")
        }.padding(.top, 6)
        TextField("Search connections", text: $search).textFieldStyle(.plain)
          .font(.system(size: 12)).padding(9).background(
            PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 8))
        PaddockScrollView {
          LazyVStack(spacing: 4) {
            ForEach(rows) { row in
              Button {
                selection = row.id
              } label: {
                VStack(alignment: .leading, spacing: 5) {
                  Text(row.name).font(.system(size: 13, weight: .medium)).lineLimit(1)
                  Text(
                    "\(row.models.count) models · \(row.hasKey ? "API key saved" : "No API key")"
                  )
                  .font(.system(size: 11)).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, alignment: .leading).padding(12)
                  .background(
                    active?.id == row.id ? PaddockStyle.elevated : PaddockStyle.canvas,
                    in: RoundedRectangle(cornerRadius: 8)
                  )
                  .contentShape(Rectangle())
              }.buttonStyle(QuietButtonStyle()).accessibilityIdentifier("connection-row-\(row.id)")
            }
          }
        }
      }.padding(16).frame(minWidth: 230, idealWidth: 280, maxWidth: 360)
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 20) {
          if let active {
            Text(active.name).font(.system(size: 24, weight: .semibold))
            Text(active.baseUrl).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(
              .enabled)
            if active.allowUnauthenticated {
              Text("No authentication").font(.system(size: 12)).foregroundStyle(.secondary)
            }
            HStack {
              Button("Configure & check") { model.review(connection: active) }.buttonStyle(
                FlatButtonStyle(primary: true))
              Button("Remove connection") { removing = active }.buttonStyle(FlatButtonStyle())
            }.disabled(model.busy)
            WorkspaceRule()
            Text("Models in Studio").font(.system(size: 14, weight: .medium))
            if active.models.isEmpty {
              Text("No models selected").font(
                .system(size: 12)
              ).foregroundStyle(.secondary)
            }
            ForEach(active.models, id: \.pickKey) { pick in
              HStack {
                VStack(alignment: .leading, spacing: 5) {
                  Text(pick.display ?? pick.id).font(.system(size: 13)).lineLimit(2)
                  Text(pick.provider.map { "Provider: \($0) · no fallback" } ?? pick.id).font(
                    .system(size: 11)
                  ).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Remove model", systemImage: "minus.circle") {
                  Task { await model.removeModel(pick, from: active) }
                }
                .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).disabled(model.busy)
              }
            }
          } else if model.loaded {
            Text("Connect your model server").font(.system(size: 24, weight: .semibold))
            Button("Add endpoint", systemImage: "plus") { model.review() }.buttonStyle(
              FlatButtonStyle(primary: true))
          } else {
            ProgressView("Loading connections…")
          }
          if let error = model.error {
            Text(error).font(.system(size: 12)).foregroundStyle(PaddockStyle.caution).textSelection(
              .enabled)
            Button("Try again") { Task { await model.refresh() } }
              .buttonStyle(FlatButtonStyle()).disabled(model.loading)
          }
        }.padding(28).frame(maxWidth: .infinity, alignment: .leading)
      }.frame(minWidth: 350, maxWidth: .infinity, maxHeight: .infinity)
    }.background(PaddockStyle.canvas).task { await model.refresh() }
      .confirmationDialog(
        "Remove this connection?",
        isPresented: Binding(get: { removing != nil }, set: { if !$0 { removing = nil } }),
        titleVisibility: .visible
      ) {
        if let row = removing {
          Button("Remove \(row.name)", role: .destructive) {
            Task { await model.remove(row) }
            removing = nil
          }
        }
        Button("Cancel", role: .cancel) { removing = nil }
      } message: {
        Text(
          "Its models will leave the composer. Saved conversations and local running models will not be deleted or stopped."
        )
      }
  }
}
