import SwiftUI

struct StudioLibraryView: View {
  @Bindable var model: StudioLibraryModel
  @State private var confirmDiscard = false
  @State private var confirmDelete = false
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 20) {
        PageHeading(title: "Prompts") {
          if model.editor == nil {
            Button("New preset", systemImage: "plus") { model.create() }.buttonStyle(
              FlatButtonStyle(primary: true))
          }
        }
        if let error = model.error {
          Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
            .accessibilityIdentifier("preset-error")
        }
        if let notice = model.notice { Text(notice).foregroundStyle(.secondary) }
        if model.editor != nil { editor } else { library }
      }.font(.system(size: 13)).padding(32).frame(maxWidth: 820).frame(maxWidth: .infinity)
    }.background(PaddockStyle.canvas)
      .task(id: model.query) {
        do {
          try await Task.sleep(for: .milliseconds(180))
          try Task.checkCancellation()
        } catch { return }
        await model.load()
      }
      .confirmationDialog(
        "Discard unsaved preset changes?", isPresented: $confirmDiscard, titleVisibility: .visible
      ) {
        Button("Discard changes", role: .destructive) {
          model.discard()
          Task { await model.load() }
        }
        Button("Keep editing", role: .cancel) {}
      }
      .confirmationDialog(
        "Delete this preset?", isPresented: $confirmDelete, titleVisibility: .visible
      ) {
        Button("Delete preset", role: .destructive) { model.remove() }
        Button("Cancel", role: .cancel) {}
      } message: {
        Text("This cannot be undone. Chats already using it keep their instructions.")
      }
  }
  private var library: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack {
        TextField("Search names and prompt text", text: $model.query).textFieldStyle(
          StudioPopoverFieldStyle()
        ).accessibilityIdentifier("preset-search")
      }
      if model.loading { ProgressView().controlSize(.small) }
      if let page = model.page {
        if page.rows.isEmpty {
          Text(
            page.total == 0
              ? "No presets yet"
              : "No presets match your search."
          ).foregroundStyle(.secondary).padding(.vertical, 24)
        }
        LazyVStack(spacing: 0) {
          ForEach(page.rows) { row in
            Button {
              Task { await model.open(row.id) }
            } label: {
              VStack(alignment: .leading, spacing: 6) {
                Text(row.name).fontWeight(.medium).lineLimit(2)
                Text(row.preview).font(.system(size: 12)).foregroundStyle(.secondary).lineLimit(3)
              }.frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 16).contentShape(
                Rectangle())
            }.buttonStyle(.plain).disabled(model.loading).accessibilityIdentifier(
              "preset-\(row.id)")
            WorkspaceRule()
          }
        }
        HStack {
          Text("\(page.matched) presets").foregroundStyle(.secondary)
          Spacer()
          Button("Previous") { Task { await model.load(page: page.page - 1) } }.disabled(
            page.page == 0 || model.loading)
          Text("\(page.page + 1)").monospacedDigit()
          Button("Next") { Task { await model.load(page: page.page + 1) } }.disabled(
            (page.page + 1) * page.pageSize >= page.matched || model.loading)
        }.buttonStyle(FlatButtonStyle())
      }
    }
  }
  private var editor: some View {
    VStack(alignment: .leading, spacing: 16) {
      TextField(
        "Preset name",
        text: Binding(get: { model.editor?.name ?? "" }, set: { model.editor?.name = $0 })
      )
      .textFieldStyle(StudioPopoverFieldStyle()).accessibilityLabel("Preset name")
      PaddockTextEditor(
        text: Binding(get: { model.editor?.body ?? "" }, set: { model.editor?.body = $0 })
      )
      .scrollContentBackground(.hidden).font(.system(size: 13)).padding(10).frame(minHeight: 280)
      .background(
        PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
      ).accessibilityLabel("Preset instructions")
      if let validation = model.validation {
        Text(validation).font(.caption).foregroundStyle(.secondary)
      }
      HStack {
        Button("Back to presets") {
          if model.dirty {
            confirmDiscard = true
          } else {
            model.discard()
            Task { await model.load() }
          }
        }
        if model.editor?.revision.isEmpty == false {
          Button("Delete…", role: .destructive) { confirmDelete = true }.disabled(model.dirty)
        }
        Spacer()
        Button(model.saving ? "Saving…" : "Save preset") { model.save() }
          .buttonStyle(FlatButtonStyle(primary: true)).disabled(
            !model.dirty || model.validation != nil)
      }.buttonStyle(FlatButtonStyle())
    }.disabled(model.saving || model.loading)
  }
}
