import PaddockStudio
import SwiftUI

/// Same writer grouping, lane order and all-or-nothing split as ArtifactPanel.vue.
/// Pane controls and source are Swift; only executable document previews use WebKit.
struct NativeArtifactPanel: View {
  @Bindable var workspace: StudioWorkspace
  private var items: [StudioState.Artifact] { workspace.presentedArtifacts }
  private var groups: [ArtifactPresentation.Group] {
    ArtifactPresentation.groups(
      items,
      laneOrder: workspace.state?.nativeTranscript?.messages.filter { $0.role == "assistant" }.map(
        \.model) ?? [])
  }
  var body: some View {
    GeometryReader { geometry in
      if ArtifactPresentation.splits(groups: groups.count, width: geometry.size.width) {
        HSplitView {
          ForEach(groups) { group in
            pane(
              items: group.items,
              selected: workspace.artifactPicks[group.model] ?? group.items[0].id
            ) { id in
              workspace.artifactPicks[group.model] = id
            }.frame(minWidth: 340, maxWidth: .infinity, maxHeight: .infinity)
          }
        }
      } else {
        pane(items: items, selected: workspace.selectedArtifactId ?? items.first?.id ?? "") {
          workspace.selectedArtifactId = $0
        }
      }
    }
    .onChange(of: workspace.selectedArtifactId, initial: true) { _, id in
      if let artifact = items.first(where: { $0.id == id }) {
        workspace.artifactPicks[artifact.model] = artifact.id
      }
    }
  }
  private func pane(
    items: [StudioState.Artifact], selected: String, pick: @escaping (String) -> Void
  ) -> some View {
    let chosen = items.first { $0.id == selected } ?? items.first
    return VStack(spacing: 0) {
      if items.count > 1 {
        PaddockScrollView(.horizontal) {
          HStack(spacing: 4) {
            ForEach(items) { artifact in
              Button {
                pick(artifact.id)
              } label: {
                HStack(spacing: 5) {
                  if let draft = workspace.artifactDrafts[artifact.id], draft.saved != draft.text {
                    Circle().frame(width: 5, height: 5)
                  }
                  Text(artifact.title).lineLimit(1)
                }.padding(.horizontal, 9).padding(.vertical, 7)
                  .background(
                    chosen?.id == artifact.id ? PaddockStyle.elevated : .clear,
                    in: RoundedRectangle(cornerRadius: 7))
              }.buttonStyle(.plain).help(artifact.title).accessibilityIdentifier(
                "artifact-tab-\(artifact.id)")
            }
          }.font(.system(size: 11)).padding(8)
        }.scrollIndicators(.hidden)
      }
      if let chosen { NativeArtifactView(artifact: chosen, workspace: workspace).id(chosen.id) }
    }.fullHeightWorkspaceColumn()
  }
}
