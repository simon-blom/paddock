import SwiftUI

/// One row per saved text/question set, not per run. Uses the same SQLite
/// documents as web Reads; each document's runs stay in the Answers panel.
struct NativeReadsSidebar: View {
  @Bindable var model: NativeReadsModel
  @State private var search = ""
  @State private var pendingOpen: NativeReadsModel.Session?
  @State private var confirmNew = false
  @State private var pendingDelete: NativeReadsModel.Session?
  @State private var renaming: String?
  @State private var title = ""
  @FocusState private var renameFocused: Bool

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      Button {
        guard !model.historyNavigationBlocked else { return }
        if model.unsavedRead { confirmNew = true } else { model.reset() }
      } label: {
        Label("New read", systemImage: "plus")
          .font(.system(size: 13, weight: .medium))
          .padding(.horizontal, 8).frame(height: 32)
          .frame(maxWidth: .infinity, alignment: .leading)
      }.buttonStyle(QuietButtonStyle()).disabled(model.historyNavigationBlocked)
        .accessibilityIdentifier("sidebar-new-read")
      HStack(spacing: 8) {
        Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
        TextField("Search reads", text: $search).textFieldStyle(.plain)
          .accessibilityIdentifier("reads-history-search")
        if !search.isEmpty {
          Button("Clear search", systemImage: "xmark") { search = "" }
            .labelStyle(.iconOnly).buttonStyle(.plain)
        }
      }.padding(10).background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 8))
      if let error = model.historyListError ?? model.historyError {
        Text(error).font(.system(size: 12)).foregroundStyle(PaddockStyle.caution)
          .textSelection(.enabled).accessibilityIdentifier("reads-history-error")
      }
      PaddockScrollView {
        LazyVStack(spacing: 4) {
          let rows = model.visibleSessions(search: search)
          if !model.historyLoaded {
            ProgressView().controlSize(.small).frame(maxWidth: .infinity).padding(.top, 40)
          } else if rows.isEmpty {
            Text(
              search.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? "No reads yet" : "No matches"
            )
            .foregroundStyle(.secondary).frame(maxWidth: .infinity).padding(.top, 40)
          }
          ForEach(rows) { session in row(session) }
        }
      }.frame(maxHeight: .infinity)
    }.padding(.horizontal, 10).padding(.vertical, 18)
      .accessibilityElement(children: .contain).accessibilityLabel("Reads")
      .accessibilityIdentifier("studio-reads-sidebar")
      .task { await model.refreshHistory() }
      .onChange(of: renameFocused) { _, focused in if !focused { commitRename() } }
      .confirmationDialog(
        "Discard changes to this read?",
        isPresented: Binding(
          get: { confirmNew || pendingOpen != nil },
          set: {
            if !$0 {
              confirmNew = false
              pendingOpen = nil
            }
          }),
        titleVisibility: .visible
      ) {
        Button("Discard changes", role: .destructive) {
          guard !model.historyNavigationBlocked else { return }
          let next = pendingOpen
          pendingOpen = nil
          confirmNew = false
          if let next { Task { await model.openSession(next.id) } } else { model.reset() }
        }
        Button("Cancel", role: .cancel) {
          confirmNew = false
          pendingOpen = nil
        }
      }
      .confirmationDialog(
        "Delete read?",
        isPresented: Binding(
          get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
        titleVisibility: .visible
      ) {
        if let session = pendingDelete {
          Button("Delete read", role: .destructive) {
            pendingDelete = nil
            Task { await model.removeSession(session.id) }
          }
        }
        Button("Cancel", role: .cancel) { pendingDelete = nil }
      } message: {
        if let session = pendingDelete {
          Text("\(session.title) and its runs will be permanently removed. This cannot be undone.")
        }
      }
  }

  private func row(_ session: NativeReadsModel.Session) -> some View {
    HStack(spacing: 4) {
      if renaming == session.id {
        TextField("Read title", text: $title).textFieldStyle(.plain)
          .focused($renameFocused).onSubmit { commitRename() }
          .onExitCommand {
            renaming = nil
            renameFocused = false
          }
          .padding(.horizontal, 8).frame(height: 34)
          .accessibilityIdentifier("reads-rename-title")
      } else {
        Button {
          guard !model.historyNavigationBlocked, model.activeSession?.id != session.id else {
            return
          }
          if model.unsavedRead {
            pendingOpen = session
          } else {
            Task { await model.openSession(session.id) }
          }
        } label: {
          HStack(spacing: 8) {
            Image(systemName: "list.bullet.clipboard").foregroundStyle(.secondary).frame(width: 14)
            Text(session.title).lineLimit(1).truncationMode(.tail)
            Spacer(minLength: 0)
            Text(Self.when(session.updatedAt)).font(.system(size: 10)).foregroundStyle(.secondary)
          }.font(.system(size: 12)).padding(.leading, 8).frame(height: 34)
            .contentShape(Rectangle())
        }.buttonStyle(QuietButtonStyle()).disabled(model.historyNavigationBlocked)
          .accessibilityIdentifier("reads-open-\(session.id)")
          .accessibilityLabel(session.title)
          .accessibilityAddTraits(model.activeSession?.id == session.id ? .isSelected : [])
          .help(
            "\(session.title) · \(session.runs) \(session.runs == 1 ? "run" : "runs") · \(session.model) · \(Date(timeIntervalSince1970: session.updatedAt / 1000).formatted())"
          )
        Menu {
          Button("Rename", systemImage: "pencil") {
            title = session.title
            renaming = session.id
            renameFocused = true
          }
          Divider()
          Button("Delete", systemImage: "trash", role: .destructive) { pendingDelete = session }
        } label: {
          Image(systemName: "ellipsis").frame(width: 26, height: 28).contentShape(Rectangle())
        }.menuStyle(.button).buttonStyle(QuietButtonStyle()).menuIndicator(.hidden)
          .fixedSize().foregroundStyle(.secondary).disabled(model.historyNavigationBlocked)
          .accessibilityLabel("Actions for \(session.title)")
          .accessibilityIdentifier("reads-actions-\(session.id)")
      }
    }.background(
      model.activeSession?.id == session.id ? PaddockStyle.elevated : .clear,
      in: RoundedRectangle(cornerRadius: 7))
  }

  private func commitRename() {
    guard let id = renaming else { return }
    renaming = nil
    renameFocused = false
    let next = title.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !next.isEmpty, next != model.sessions.first(where: { $0.id == id })?.title else { return }
    Task { await model.renameSession(id, title: next) }
  }

  private static func when(_ milliseconds: Double) -> String {
    let date = Date(timeIntervalSince1970: milliseconds / 1000)
    let age = max(0, Date().timeIntervalSince(date))
    if age < 60 { return "now" }
    if age < 3600 { return "\(Int(age / 60))m" }
    if age < 86400 { return "\(Int(age / 3600))h" }
    if age < 604800 { return "\(Int(age / 86400))d" }
    return date.formatted(.dateTime.month(.abbreviated).day())
  }
}
