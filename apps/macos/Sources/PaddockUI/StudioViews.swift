import SwiftUI

struct StudioProjectsView: View {
  @State private var newProject = false
  var body: some View {
    WorkspacePlaceholder(
      title: "Projects", subtitle: "Keep related chats, files, and instructions together.",
      symbol: "folder", heading: "Make room for your next idea",
      detail: "Project storage is not connected yet. You can explore the setup for a new project.",
      action: "New project", onAction: { newProject = true }
    ).sheet(isPresented: $newProject) {
      ProjectSetupView().presentationBackground(PaddockStyle.canvas)
    }
  }
}

struct ProjectSetupView: View {
  @Environment(\.dismiss) private var dismiss
  @State private var name = ""
  @State private var instructions = ""
  var body: some View {
    VStack(alignment: .leading, spacing: 22) {
      Text("New project").font(.system(size: 22, weight: .semibold))
      VStack(alignment: .leading, spacing: 8) {
        Text("Name").font(.system(size: 12, weight: .medium))
        TextField("What are you working on?", text: $name).textFieldStyle(.plain)
          .padding(12).background(
            PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
          )
          .accessibilityLabel("Project name")
      }
      VStack(alignment: .leading, spacing: 8) {
        Text("Instructions").font(.system(size: 12, weight: .medium))
        PaddockTextEditor(text: $instructions).scrollContentBackground(.hidden).frame(height: 110)
          .padding(8).background(
            PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card)
          )
          .accessibilityLabel("Project instructions")
      }
      Label("Project saving is not connected yet.", systemImage: "info.circle")
        .font(.system(size: 12)).foregroundStyle(.secondary)
      HStack {
        Spacer()
        Button("Cancel") { dismiss() }.buttonStyle(FlatButtonStyle())
          .keyboardShortcut(.cancelAction)
        Button("Create project") {}.modifier(PrimaryAction()).disabled(true)
      }
    }.padding(28).frame(width: 520).background(PaddockStyle.canvas)
      .tint(PaddockStyle.accent)
  }
}

/// Explicit empty surfaces, never sample records or fictitious progress. This
/// keeps the complete navigation reviewable before the storage/jobs bridge lands.
struct WorkspacePlaceholder: View {
  let title: String
  let subtitle: String
  let symbol: String
  let heading: String
  let detail: String
  let action: String
  let onAction: () -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      PageHeading(title: title, subtitle: subtitle) { EmptyView() }.padding(.top, 28)
      Spacer(minLength: 32)
      VStack(spacing: 16) {
        Image(systemName: symbol).font(.system(size: 28, weight: .light))
          .foregroundStyle(.secondary)
        Text(heading).font(.system(size: 20, weight: .medium))
        Text(detail).font(.system(size: 13)).foregroundStyle(.secondary)
          .multilineTextAlignment(.center).lineSpacing(4).frame(maxWidth: 390)
        Button(action, action: onAction).buttonStyle(FlatButtonStyle()).padding(.top, 6)
      }.frame(maxWidth: .infinity)
      Spacer(minLength: 60)
    }.padding(.horizontal, 32).padding(.bottom, 32)
      .frame(maxWidth: 960, maxHeight: .infinity)
      .frame(maxWidth: .infinity, maxHeight: .infinity).background(PaddockStyle.canvas)
  }
}
