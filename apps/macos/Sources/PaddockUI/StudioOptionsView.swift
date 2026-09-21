import PaddockStudio
import SwiftUI

struct StudioCompareView: View {
  @Bindable var chat: StudioWorkspace
  @Environment(\.dismiss) private var dismiss
  @State private var selected = Set<String>()
  @State private var search = ""
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      StudioPopoverHeading(title: "Compare models")
      TextField("Search models or providers", text: $search).textFieldStyle(
        StudioPopoverFieldStyle())
      StudioCompareSelectionBar(selected: $selected, busy: chat.busy)
      StudioCompareList(models: chat.state?.models ?? [], selected: $selected, search: search)
      if let error = chat.error { Text(error).foregroundStyle(.secondary) }
      HStack {
        Spacer()
        Button("Cancel") { dismiss() }.buttonStyle(FlatButtonStyle())
        Button("Use selection") {
          Task {
            let existing = (chat.state?.selectedModels ?? []).filter { selected.contains($0) }
            let added = selected.subtracting(existing).sorted()
            await chat.perform(
              "models", ["ids": .array((existing + added).map(StudioValue.string))])
            if chat.error == nil { dismiss() }
          }
        }.buttonStyle(FlatButtonStyle(primary: true))
          .disabled(
            !StudioCompareList.canApply(selected, models: chat.state?.models ?? []) || chat.busy)
      }
    }.font(.system(size: 12)).padding(16).frame(width: 380)
      .task { selected = Set(chat.state?.selectedModels ?? []) }
  }
}

/// Edits only the popover draft, independently of the catalog's search filter.
struct StudioCompareSelectionBar: View {
  @Binding var selected: Set<String>
  var busy = false

  func clearSelection() {
    guard !busy else { return }
    selected.removeAll()
  }

  var body: some View {
    HStack {
      Text(selected.isEmpty ? "No models selected" : "\(selected.count) selected")
        .font(.system(size: 11)).foregroundStyle(.secondary)
        .accessibilityIdentifier("compare-selection-count")
      Spacer()
      Button("Clear selection", action: clearSelection)
        .buttonStyle(FlatButtonStyle()).disabled(selected.isEmpty || busy)
        .accessibilityIdentifier("compare-clear-selection")
        .help(
          "Clear all selected models, including ones hidden by search. "
            + "Nothing is applied until you use the selection.")
    }
  }
}

/// An intrinsically bounded list: a maxHeight-only ScrollView can collapse to
/// one checkbox in an NSPopover even when more rows exist in its AX tree.
/// Row/group geometry sets the viewport; larger libraries scroll normally.
struct StudioCompareList: View {
  let models: [StudioState.Model]
  @Binding var selected: Set<String>
  var search = ""

  struct Group: Identifiable {
    let id: String
    let title: String
    let models: [StudioState.Model]
  }

  static func groups(_ models: [StudioState.Model], search: String) -> [Group] {
    let terms = search.split(whereSeparator: \.isWhitespace).map(String.init)
    let available = models.filter { model in
      model.status == "ok"
        && terms.allSatisfy { term in
          [model.title, model.id, model.provider, model.vendor].contains {
            $0.localizedCaseInsensitiveContains(term)
          }
        }
    }
    var groups: [Group] = []
    let local = available.filter { $0.port != nil }
    if !local.isEmpty { groups.append(Group(id: "local", title: "Local", models: local)) }
    var providers: [String] = []
    for model in available where model.port == nil {
      if !providers.contains(model.provider) { providers.append(model.provider) }
    }
    for provider in providers {
      groups.append(
        Group(
          id: "cloud:\(provider)", title: provider,
          models: available.filter {
            $0.port == nil && $0.provider == provider
          }))
    }
    return groups
  }

  static func canApply(_ selected: Set<String>, models: [StudioState.Model]) -> Bool {
    let choices = models.filter { selected.contains($0.id) && $0.status == "ok" }
    return (1...4).contains(selected.count) && choices.count == selected.count
      && (choices.allSatisfy(\.chat) || choices.allSatisfy(\.audio))
  }

  private var groups: [Group] { Self.groups(models, search: search) }
  private var height: CGFloat {
    min(300, max(72, CGFloat(groups.reduce(0) { $0 + 26 + $1.models.count * 52 + 8 })))
  }
  var body: some View {
    PaddockScrollView {
      LazyVStack(alignment: .leading, spacing: 0) {
        if groups.isEmpty {
          Text(
            search.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              ? "No available models. Start a local model or add a cloud model in Settings."
              : "No models match your search."
          )
          .font(.system(size: 12)).foregroundStyle(.secondary)
          .frame(maxWidth: .infinity, minHeight: 72, alignment: .leading)
        }
        ForEach(groups) { group in
          Text(group.title).font(.system(size: 11, weight: .semibold)).foregroundStyle(.secondary)
            .frame(height: 26).accessibilityAddTraits(.isHeader)
          ForEach(group.models) { model in
            StudioCompareModelRow(
              model: model,
              isSelected: Binding(
                get: { selected.contains(model.id) },
                set: { enabled in
                  if enabled { selected.insert(model.id) } else { selected.remove(model.id) }
                })
            )
            .disabled(
              !selected.contains(model.id)
                && !Self.canApply(selected.union([model.id]), models: models)
            )
          }
          Color.clear.frame(height: 8).accessibilityHidden(true)
        }
      }.frame(maxWidth: .infinity, alignment: .leading)
    }.frame(height: height).accessibilityIdentifier("compare-model-list")
  }
}

/// A checkbox's default compound label aligns to the first text baseline.
/// Lay out the native control separately so its center, the maker mark and
/// the two-line identity share one axis. The label remains a click target,
/// but only the checkbox participates in keyboard/VoiceOver navigation.
struct StudioCompareModelRow: View {
  let model: StudioState.Model
  @Binding var isSelected: Bool

  var body: some View {
    HStack(alignment: .center, spacing: 9) {
      Toggle("Select model", isOn: $isSelected)
        .toggleStyle(.checkbox).labelsHidden()
        .accessibilityLabel("\(model.title) · \(model.provider)")
        .accessibilityIdentifier("compare-model-\(model.id)")
      Button {
        isSelected.toggle()
      } label: {
        HStack(alignment: .center, spacing: 9) {
          ModelAvatar(vendor: model.vendor, size: 26)
          VStack(alignment: .leading, spacing: 4) {
            Text(model.title).font(.system(size: 12, weight: .medium)).lineLimit(1)
            Text(model.port.map { "Port \(String($0))" } ?? model.provider)
              .font(.system(size: 11)).foregroundStyle(.secondary).lineLimit(1)
          }
          Spacer(minLength: 0)
        }.frame(maxWidth: .infinity, alignment: .leading)
          .contentShape(Rectangle())
      }.buttonStyle(.plain).focusable(false).accessibilityHidden(true)
    }.frame(height: 52)
      .help("\(model.title) · \(model.provider)")
  }
}
