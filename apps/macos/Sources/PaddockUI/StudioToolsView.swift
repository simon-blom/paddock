import PaddockStudio
import SwiftUI

extension EnvironmentValues {
  @Entry var studioToolsManager: IntegrationsModel? = nil
}

/// Swift presents controls; the same reducer as the web composer owns selection.
/// There is no second editable selection, Apply button, or native All-mode rule.
struct StudioToolsView: View {
  @Bindable var chat: StudioWorkspace
  @Environment(\.dismiss) private var dismiss
  @Environment(\.studioToolsManager) private var manager
  @State private var search = ""
  @State private var expanded = Set<String>()
  @State private var saving = false
  private var groups: [StudioState.ToolGroup] { chat.state?.tools ?? [] }
  private var all: Bool { chat.state?.settings["toolSelection"]?.object?["mode"]?.text != "custom" }
  private var count: Int {
    chat.state?.settings["toolSelection"]?.object?["picks"]?.array?.count ?? 0
  }
  private var listHeight: CGFloat {
    let rows = groups.reduce(0) { $0 + 1 + (isExpanded($1) ? max(1, $1.tools.count) : 0) }
    return min(320, max(100, CGFloat(rows) * 42))
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      StudioPopoverHeading(title: "Tools and connectors")
        .padding(.horizontal, 16).padding(.top, 16).padding(.bottom, 12)
      HStack(spacing: 8) {
        Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
        TextField("Search tools or connectors", text: $search).textFieldStyle(.plain)
          .accessibilityLabel("Search tools or connectors")
          .accessibilityIdentifier("studio-tools-search")
        if !search.isEmpty {
          Button("Clear search", systemImage: "xmark") { search = "" }
            .labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
        }
      }.padding(10).background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: 6))
        .padding(.horizontal, 12).padding(.bottom, 8)
      StudioPopoverChoice(
        title: "All tools", selected: all
      ) {
        choose("all")
      }.padding(.horizontal, 7).accessibilityIdentifier("studio-tools-all")
      WorkspaceRule().padding(.top, 6)
      PaddockScrollView {
        LazyVStack(alignment: .leading, spacing: 0) {
          if groups.isEmpty {
            Text(search.isEmpty ? "No tools available for this model." : "No matching tools")
              .foregroundStyle(.secondary).padding(16)
          }
          ForEach(groups) { group in
            HStack(spacing: 0) {
              Button {
                if expanded.contains(group.id) {
                  expanded.remove(group.id)
                } else {
                  expanded.insert(group.id)
                }
              } label: {
                Image(systemName: isExpanded(group) ? "chevron.down" : "chevron.right")
                  .font(.system(size: 9, weight: .semibold)).frame(width: 28, height: 40)
                  .contentShape(Rectangle())
              }.buttonStyle(QuietButtonStyle()).disabled(!search.isEmpty)
                .accessibilityLabel("\(isExpanded(group) ? "Collapse" : "Expand") \(group.label)")
              Button {
                choose("group", group: group)
              } label: {
                HStack(spacing: 8) {
                  Text(group.label).font(.system(size: 12, weight: .medium)).lineLimit(1)
                  if !group.connectorId.isEmpty {
                    Text("Connector").font(.system(size: 10)).foregroundStyle(.secondary)
                  }
                  Spacer(minLength: 0)
                  Text("\(group.selectedCount ?? 0)/\(group.total ?? group.tools.count)")
                    .monospacedDigit().foregroundStyle(.secondary)
                  Image(systemName: group.checked == "some" ? "minus" : "checkmark")
                    .frame(width: 14).opacity(
                      group.checked == "all" || group.checked == "some" ? 1 : 0)
                }.padding(.trailing, 10).frame(height: 40).contentShape(Rectangle())
              }.buttonStyle(QuietButtonStyle()).accessibilityLabel("Select \(group.label)")
                .accessibilityValue(group.checked ?? "none")
            }.padding(.horizontal, 7)
            if isExpanded(group) {
              if group.status == "loading" {
                Text("Loading tools…").foregroundStyle(.secondary).padding(.leading, 36)
              } else if group.status == "error" {
                HStack {
                  Text("Could not list tools. The server can still be selected.")
                  Button("Retry") { Task { await chat.perform("tools") } }.buttonStyle(
                    QuietButtonStyle())
                }.foregroundStyle(.secondary).padding(.horizontal, 16)
              } else if group.tools.isEmpty {
                Text("No tools available").foregroundStyle(.secondary).padding(.leading, 36)
              }
              ForEach(group.tools) { tool in
                StudioPopoverChoice(
                  title: tool.name, subtitle: tool.description, selected: tool.selected == true
                ) {
                  choose("tool", group: group, tool: tool.name)
                }.help(tool.description ?? tool.name).padding(.leading, 26).padding(.trailing, 7)
              }
            }
          }
        }
      }.frame(height: listHeight)
      if let error = chat.error {
        Text(error).foregroundStyle(PaddockStyle.caution).padding(12)
      }
      WorkspaceRule()
      HStack {
        Text(all ? "All tools" : "\(count) selected").foregroundStyle(.secondary)
        Spacer()
        if let manager {
          Button("Manage connectors…") {
            dismiss()
            manager.onManage?()
          }
          .buttonStyle(QuietButtonStyle())
        }
      }.padding(12)
    }.font(.system(size: 11)).frame(width: 380).studioPopoverSurface()
      .disabled(saving || chat.busy)
      .task { await chat.perform("tools") }
      .task(id: search) {
        do { try await Task.sleep(for: .milliseconds(120)) } catch { return }
        await chat.perform("toolQuery", ["query": .string(search)])
      }
  }
  private func isExpanded(_ group: StudioState.ToolGroup) -> Bool {
    !search.isEmpty || expanded.contains(group.id)
  }
  private func choose(_ action: String, group: StudioState.ToolGroup? = nil, tool: String? = nil) {
    guard !saving else { return }
    saving = true
    var payload: [String: StudioValue] = ["action": .string(action)]
    if let group { payload["label"] = .string(group.id) }
    if let tool { payload["tool"] = .string(tool) }
    Task {
      await chat.perform("toolPicker", payload)
      saving = false
    }
  }
}
