import PaddockStudio
import SwiftUI

struct StudioHeaderModelView: View {
  @Bindable var chat: StudioWorkspace

  var body: some View {
    if let header = chat.state?.modelHeader {
      StudioHeaderModelContent(
        header: header, enabled: chat.ready && !chat.busy && !chat.hasMessageEdit
      ) { id in
        Task { await chat.perform("models", ["ids": .array([.string(id)])]) }
      }
    }
  }
}

/// No compare mutation here: like the web, its lane picker remains in the
/// composer. The full identities stay accessible when narrow titles truncate.
struct StudioHeaderModelContent: View {
  let header: StudioModelHeader
  var enabled: Bool
  var select: (String) -> Void

  var body: some View {
    Group {
      if header.comparing {
        comparisonHeader
      } else if let current = header.current {
        HStack(spacing: 7) {
          Menu {
            ForEach(header.pickerOptions) { option in
              Button {
                select(option.value)
              } label: {
                if option.value == header.currentModel {
                  Label("\(option.label) · \(option.hint)", systemImage: "checkmark")
                } else {
                  Text("\(option.label) · \(option.hint)")
                }
              }.disabled(!option.available).help(option.title)
            }
          } label: {
            HStack(spacing: 6) {
              vendorMark(current.vendor)
              Text(current.label).lineLimit(1).truncationMode(.middle)
              if !current.available {
                Image(systemName: "exclamationmark.circle").help("Model is not running")
              }
              Image(systemName: "chevron.down").font(.system(size: 9, weight: .semibold))
            }
          }.menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
            .fixedSize(horizontal: false, vertical: true)
            .disabled(!enabled).help(current.title)
            .accessibilityLabel("Studio model")
            .accessibilityValue(current.label + (current.available ? "" : " · not running"))
            .accessibilityIdentifier("studio-header-model")
          if header.isVision {
            Image(systemName: "eye").foregroundStyle(.secondary)
              .help("This model can read images").accessibilityLabel("Supports vision")
          }
          specBadge(header.specLabel)
        }
      } else if let encoder = header.soleEncoder {
        // The Embeddings page is not native yet. Do not fabricate a working
        // navigation link or put an encoder in a conversational model picker.
        Text(encoder).lineLimit(1).truncationMode(.middle)
          .help(
            "Embeddings and rerank require the web Studio; the native page is not available yet.")
      }
    }.font(.system(size: 12, weight: .medium)).foregroundStyle(.primary)
      .frame(maxWidth: 280)
  }

  private var comparisonHeader: some View {
    comparisonLanes
      .accessibilityElement(children: .ignore)
      .accessibilityLabel("Comparing models")
      .accessibilityValue(comparisonDescription)
      .accessibilityIdentifier("studio-header-compare")
      .help(comparisonLabels)
  }

  private var comparisonLabels: String {
    header.compareLanes.map(\.label).joined(separator: " vs. ")
  }

  private var comparisonDescription: String {
    header.compareLanes.map { $0.label + " " + $0.spec }.joined(separator: " vs. ")
  }

  private var comparisonLanes: some View {
    HStack(spacing: 6) {
      ForEach(header.compareLanes) { lane in
        if lane.id != header.compareLanes.first?.id {
          Text("vs.").foregroundStyle(.secondary)
        }
        compareLane(lane)
      }
    }
  }

  private func compareLane(_ lane: StudioModelHeader.Lane) -> some View {
    HStack(spacing: 5) {
      vendorMark(lane.vendor)
      Text(lane.label).lineLimit(1).truncationMode(.middle)
      specBadge(lane.spec)
    }.help(lane.id)
  }

  @ViewBuilder private func vendorMark(_ vendor: String) -> some View {
    if let icon = ProviderArtwork.image(for: vendor) {
      Image(nsImage: icon).renderingMode(.template).resizable().scaledToFit()
        .frame(width: 14, height: 14).accessibilityHidden(true)
    }
  }

  @ViewBuilder private func specBadge(_ spec: String) -> some View {
    if !spec.isEmpty {
      Text(spec).font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
        .help("Speculative decode: \(spec)").accessibilityLabel("Speculative decode: \(spec)")
    }
  }
}
