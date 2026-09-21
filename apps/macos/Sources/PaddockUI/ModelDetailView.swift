import PaddockClient
import SwiftUI

/// The right browser column. The parent owns export identity so refreshes and
/// filtering cannot leave a stale model header paired with another model's weights.
struct ModelDetailView: View {
  let model: CatalogModel
  let backend: String?
  let canStart: Bool
  let onStart: ((String, String) -> Void)?
  var onDownload: ((String, String) -> Void)? = nil
  var purpose: ModelStartPurpose = .all
  @Binding var artifactID: String?

  private var choices: [CatalogArtifact] { purpose.weights(model, backend: backend) }
  private var selected: CatalogArtifact? {
    choices.first { $0.id == artifactID }
      ?? LibraryEntry(model: model, artifacts: choices).preferredArtifact
  }
  private var companions: [CatalogArtifact] {
    LibraryCatalog.companions(model: model, artifact: selected, backend: backend)
  }

  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 26) {
        header
        if let selected {
          downloadOptions(selected)
          exportDetails(selected)
          about(selected)
          source(selected)
        }
      }.padding(.horizontal, 22).padding(.top, 24).padding(.bottom, 32)
        .frame(maxWidth: 840, alignment: .leading)
        .frame(maxWidth: .infinity, alignment: .top)

    }.background(PaddockStyle.canvas).accessibilityIdentifier("model-detail")
  }

  private var header: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack(spacing: 14) {
        ModelAvatar(vendor: model.vendor, size: 44)
        VStack(alignment: .leading, spacing: 6) {
          Text(model.display).font(.system(size: 23, weight: .semibold)).tracking(-0.5)
            .textSelection(.enabled).accessibilityIdentifier("detail-model-name")
          Text(model.vendor ?? model.id).foregroundStyle(.secondary).font(.system(size: 12))
        }
      }
      if let url = LibraryCatalog.webURL(model.specs?.homepage) {
        Link(destination: url) { Label("Model page", systemImage: "arrow.up.right") }
          .font(.system(size: 12)).buttonStyle(FlatButtonStyle())
      }
    }
  }

  private func downloadOptions(_ selected: CatalogArtifact) -> some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack {
        sectionTitle("Download options")
        Spacer()
        Text(backend == "metal" ? "For macOS · Metal" : (backend ?? "Backend unavailable"))
          .font(.system(size: 10)).foregroundStyle(.secondary)
      }
      VStack(spacing: 4) {
        ForEach(choices) { artifact in
          Button {
            artifactID = artifact.id
          } label: {
            HStack(alignment: .top, spacing: 10) {
              Image(systemName: selected.id == artifact.id ? "largecircle.fill.circle" : "circle")
                .font(.system(size: 13)).foregroundStyle(.secondary).padding(.top, 2)
              VStack(alignment: .leading, spacing: 6) {
                HStack(alignment: .firstTextBaseline) {
                  Text(artifact.shortFormat).font(.system(size: 12, weight: .medium))
                    .fixedSize(horizontal: false, vertical: true)
                  Spacer(minLength: 6)
                  Text(DisplayFormat.bytes(artifact.totalSize)).font(.system(size: 11))
                    .monospacedDigit().foregroundStyle(.secondary).fixedSize()
                }
                Text(artifact.displayLabel).font(.system(size: 11)).foregroundStyle(.secondary)
                  .fixedSize(horizontal: false, vertical: true)
                ViewThatFits(in: .horizontal) {
                  HStack(spacing: 8) {
                    optionStatus(artifact)
                    defaultLabel(artifact)
                  }
                  VStack(alignment: .leading, spacing: 6) {
                    optionStatus(artifact)
                    defaultLabel(artifact)
                  }
                }
              }
            }.padding(12).frame(maxWidth: .infinity, alignment: .leading)
              .background(
                selected.id == artifact.id ? PaddockStyle.elevated : PaddockStyle.surface,
                in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
          }.buttonStyle(.plain).accessibilityAddTraits(
            selected.id == artifact.id ? .isSelected : []
          )
          .accessibilityLabel(
            "\(artifact.shortFormat), \(DisplayFormat.bytes(artifact.totalSize)), \(artifact.installed ? "downloaded" : "not downloaded"), \(artifact.supportNotice ?? ""), \(LibraryRecommendation.label(for: artifact) ?? "alternative")."
          ).accessibilityIdentifier("choose-export-\(artifact.id)")
        }
      }
      VStack(alignment: .leading, spacing: 6) {
        Text("Recommendation").font(.system(size: 11, weight: .medium))
        Text(LibraryRecommendation.explanation(model: model, backend: backend, purpose: purpose))
          .font(.system(size: 11)).foregroundStyle(.secondary).lineSpacing(3)
      }
      if !companions.isEmpty {
        DisclosureGroup("Companion files · \(companions.count)") {
          VStack(alignment: .leading, spacing: 12) {
            ForEach(companions) { artifact in
              VStack(alignment: .leading, spacing: 4) {
                Text(artifact.displayLabel).font(.system(size: 12, weight: .medium))
                Text(
                  "\(DisplayFormat.bytes(artifact.totalSize)) · \(artifact.installed ? "Downloaded" : "Not downloaded")"
                )
                .font(.system(size: 11)).foregroundStyle(.secondary)
                if !artifact.installed {
                  Button("Download", systemImage: "arrow.down") {
                    onDownload?(model.id, selected.id)
                  }
                  .buttonStyle(QuietButtonStyle()).disabled(onDownload == nil)
                  .accessibilityIdentifier("download-companion-\(artifact.id)")
                }
              }
            }
            Text("Companion sizes are separate from the weights option above.")
              .font(.system(size: 11)).foregroundStyle(.secondary)
          }.frame(maxWidth: .infinity, alignment: .leading).padding(.top, 10)
        }.font(.system(size: 12))
      }
      HStack {
        Spacer()
        if LibraryCatalog.canConfigure(model: model, artifact: selected, backend: backend) {
          Button("Configure instance", systemImage: "plus") { onStart?(model.id, selected.id) }
            .modifier(PrimaryAction())
            .disabled(
              !canStart || onStart == nil
            )
            .accessibilityIdentifier("detail-start-model")
        } else {
          Button("Download \(DisplayFormat.bytes(selected.totalSize))", systemImage: "arrow.down") {
            onDownload?(model.id, selected.id)
          }
          .modifier(PrimaryAction()).disabled(
            onDownload == nil
          )
          .help("Review weights and companion files before downloading.")
          .accessibilityIdentifier("detail-download-model")
        }
      }
    }
  }

  private func optionStatus(_ artifact: CatalogArtifact) -> some View {
    Label(
      artifact.installed ? "Downloaded" : "Not downloaded",
      systemImage: artifact.installed ? "checkmark.circle" : "arrow.down.circle"
    ).font(.system(size: 10)).foregroundStyle(.secondary)
  }

  @ViewBuilder private func defaultLabel(_ artifact: CatalogArtifact) -> some View {
    if let label = LibraryRecommendation.label(for: artifact) {
      Text(label).font(.system(size: 10, weight: .medium))
        .padding(.horizontal, 6).padding(.vertical, 3)
        .background(
          PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small))
    }
  }

  private func exportDetails(_ selected: CatalogArtifact) -> some View {
    VStack(alignment: .leading, spacing: 14) {
      WorkspaceRule()
      ViewThatFits(in: .horizontal) {
        HStack {
          sectionTitle("This export")
          Spacer()
          if let notice = selected.supportNotice { StatusBadge(title: notice, tone: .caution) }
        }
        VStack(alignment: .leading, spacing: 8) {
          sectionTitle("This export")
          if let notice = selected.supportNotice { StatusBadge(title: notice, tone: .caution) }
        }
      }
      CapabilityLine(capabilities: selected.runtime?.capability ?? model.capability)
      if let context = selected.runtime?.defaultMaxCtx {
        FactRow(title: "Default context", value: "\(context) tokens").font(.system(size: 12))
      }
      if let batch = selected.runtime?.defaultMaxBatch {
        FactRow(title: "Default concurrency", value: String(batch)).font(.system(size: 12))
      }
    }
  }

  private func about(_ selected: CatalogArtifact) -> some View {
    VStack(alignment: .leading, spacing: 16) {
      WorkspaceRule()
      sectionTitle("About")
      Text(model.specs?.about ?? "No model description is published in the catalog.")
        .font(.system(size: 13)).foregroundStyle(.secondary).lineSpacing(4)
      VStack(spacing: 12) {
        FactRow(title: "Published", value: LibraryCatalog.publicationDate(model) ?? "Not reported")
        if let params = model.specs?.params { FactRow(title: "Parameters", value: params) }
        if let family = model.family { FactRow(title: "Family", value: family) }
        if let context = model.specs?.context { FactRow(title: "Model context", value: context) }
        FactRow(title: "Selected format", value: selected.shortFormat)
        FactRow(title: "Model license", value: model.license ?? "Not reported")
      }.font(.system(size: 12))
      if LibraryCatalog.publicationDate(model) != nil,
        let url = LibraryCatalog.webURL(model.specs?.publishedSource)
      {
        Link("Publication source", destination: url).font(.system(size: 11))
      }
      if let strengths = model.specs?.strengths, !strengths.isEmpty {
        specNotes("Catalog highlights", notes: strengths)
      }
      if let tradeoffs = model.specs?.tradeoffs, !tradeoffs.isEmpty {
        specNotes("Considerations", notes: tradeoffs)
      }
    }
  }

  private func source(_ selected: CatalogArtifact) -> some View {
    VStack(alignment: .leading, spacing: 14) {
      WorkspaceRule()
      sectionTitle("Source & license")
      if let source = selected.source {
        Text(source.repo).font(.system(size: 12)).foregroundStyle(.secondary)
          .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
        FactRow(title: "Export license", value: source.license ?? model.license ?? "Not reported")
          .font(.system(size: 12))
        DisclosureGroup("Pinned revision") {
          Text(source.revision).font(.system(size: 11, design: .monospaced))
            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(
              .top, 8)
        }.font(.system(size: 12)).foregroundStyle(.secondary)
      } else {
        Text(
          "Export provenance is not published in this catalog entry. See the model page for upstream details."
        )
        .font(.system(size: 12)).foregroundStyle(.secondary).lineSpacing(3)
      }
    }
  }

  private func sectionTitle(_ title: String) -> some View {
    Text(title).font(.system(size: 14, weight: .semibold))
  }

  private func specNotes(_ title: String, notes: [String]) -> some View {
    DisclosureGroup(title) {
      VStack(alignment: .leading, spacing: 10) {
        ForEach(Array(notes.enumerated()), id: \.offset) { _, note in
          Text("• \(note)").font(.system(size: 12)).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
      }.frame(maxWidth: .infinity, alignment: .leading).padding(.top, 10)
    }.font(.system(size: 12))
  }
}
