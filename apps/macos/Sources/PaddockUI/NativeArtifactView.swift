import AppKit
import PaddockNativeMarkdown
import PaddockStudio
import SwiftUI

/// Native artifact chrome, editing and non-HTML rendering. The executable
/// HTML/SVG surface alone uses a credential-free, isolated preview session.
struct NativeArtifactView: View {
  let artifact: StudioState.Artifact
  let workspace: StudioWorkspace
  @State private var content: StudioWorkspace.ArtifactContent?
  @State private var version = 0
  @State private var source = false
  @State private var error: String?
  @State private var saving = false
  @State private var copied = false
  // StateObject's deferred constructor avoids allocating a discarded AppKit
  // editor every time the parent recreates this SwiftUI value during streaming.
  @StateObject private var sourceSession = ArtifactSourceSession()
  private var draft: StudioWorkspace.ArtifactDraft? { workspace.artifactDrafts[artifact.id] }
  private var dirty: Bool { draft.map { $0.text != $0.saved } ?? false }
  private var bodyText: String {
    version == 0 ? draft?.text ?? content?.body ?? "" : content?.body ?? ""
  }
  private var language: String { ArtifactPresentation.language(artifact) }
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      let writer = ArtifactPresentation.identity(artifact, state: workspace.state)
      NativeArtifactHeader(
        writer: writer, model: artifact.model, title: artifact.title,
        source: $source, version: $version, versions: content?.versions ?? [],
        dirty: dirty, saving: saving, available: content != nil, copied: copied,
        save: save, revert: { workspace.artifactDrafts[artifact.id] = nil },
        copy: {
          NSPasteboard.general.clearContents()
          NSPasteboard.general.setString(bodyText, forType: .string)
          copied = true
        },
        download: {
          let text = bodyText
          Task { await workspace.saveText(text, name: ArtifactPresentation.filename(artifact)) }
        }, close: { workspace.dismissArtifact(artifact.id) })
      if let error {
        Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled).padding(10)
      }
      if let content {
        if source {
          ArtifactSourceEditor(
            session: sourceSession, identity: "\(artifact.id)-\(version)",
            text: Binding(
              get: { bodyText },
              set: { value in
                guard version == 0 else { return }
                if value == content.body {
                  workspace.artifactDrafts[artifact.id] = nil
                } else {
                  var next = draft ?? .init(saved: content.body)
                  next.text = value
                  workspace.artifactDrafts[artifact.id] = next
                }
              }), language: language, readOnly: version != 0,
            save: { if dirty && version == 0 { save() } })
        } else {
          preview
        }
      } else if error == nil {
        ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
      } else {
        Spacer()
      }
    }.font(.system(size: 12)).frame(
      maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading
    )
    .background(PaddockStyle.canvas).accessibilityIdentifier("native-artifact")
    .task(id: "\(artifact.id)-\(artifact.updatedAt)-\(version)") {
      content = nil
      error = nil
      do {
        let value = try await workspace.artifactContent(artifact.id, version: version)
        try Task.checkCancellation()
        content = value
      } catch is CancellationError {} catch { self.error = error.localizedDescription }
    }
    .task(id: copied) {
      guard copied else { return }
      do { try await Task.sleep(for: .seconds(1.2)) } catch { return }
      copied = false
    }
  }
  private func save() {
    guard !saving else { return }
    saving = true
    error = nil
    Task {
      defer { saving = false }
      do { try await workspace.saveArtifact(artifact.id) } catch {
        self.error = error.localizedDescription
      }
    }
  }
  @ViewBuilder private var preview: some View {
    switch artifact.kind {
    case "html", "svg":
      if let origin = workspace.artifactPreviewOrigin {
        NativeHTMLArtifact(html: bodyText, origin: origin)
      } else {
        Text("The local preview service is unavailable. Source and export still work.")
          .foregroundStyle(.secondary)
      }
    case "markdown": PaddockScrollView { NativeMarkdown(bodyText).padding(14) }
    case "mermaid":
      PaddockScrollView {
        NativeMarkdown(ArtifactPresentation.fence(bodyText, language: "mermaid")).padding(14)
      }
    case "csv": NativeCSVArtifact(source: bodyText)
    case "text":
      PaddockScrollView {
        Text(verbatim: bodyText).textSelection(.enabled).frame(
          maxWidth: .infinity, alignment: .leading)
      }
    case "graph":
      Button("Open in Traverse", systemImage: "point.3.connected.trianglepath.dotted") {
        Task { await workspace.perform("graphArtifact", ["id": .string(artifact.id)]) }
      }.buttonStyle(FlatButtonStyle()).disabled(version != 0 || dirty)
      if version != 0 || dirty {
        Text("Open the saved latest version in Traverse. Source remains available here.").font(
          .caption
        ).foregroundStyle(.secondary)
      }
      PaddockScrollView { NativeCodeBlock(code: bodyText, language: language, streaming: false) }
    default:
      PaddockScrollView { NativeCodeBlock(code: bodyText, language: language, streaming: false) }
    }
  }
}

/// ArtifactPane.vue control order. No second toolbar, always-visible version
/// picker, status narration, or native-only Stop/Reload controls.
struct NativeArtifactHeader: View {
  let writer: (name: String, vendor: String)
  let model: String
  let title: String
  @Binding var source: Bool
  @Binding var version: Int
  let versions: [StudioWorkspace.ArtifactContent.Version]
  let dirty: Bool, saving: Bool, available: Bool, copied: Bool
  var save: () -> Void
  var revert: () -> Void
  var copy: () -> Void
  var download: () -> Void
  var close: () -> Void
  var body: some View {
    HStack(spacing: 8) {
      if !writer.name.isEmpty {
        HStack(spacing: 5) {
          if !writer.vendor.isEmpty { NativeVendorMark(vendor: writer.vendor, size: 13) }
          Text(writer.name).lineLimit(1).truncationMode(.tail)
        }.font(.system(size: 12)).foregroundStyle(.secondary)
          .padding(.vertical, 2).padding(.horizontal, 7).frame(maxWidth: 190)
          .background(
            PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small)
          )
          .help("Written by \(model)").layoutPriority(-1)
      }
      Text(title).font(.system(size: 13, weight: .medium)).lineLimit(1)
        .truncationMode(.tail).help(title).layoutPriority(-1)
      if versions.count > 1 {
        Menu {
          Button("v\(versions.last?.seq ?? 1) · latest") { version = 0 }
          ForEach(versions.reversed()) { item in
            if item.seq != versions.last?.seq {
              Button("v\(item.seq) · \(item.op)") { version = item.seq }
            }
          }
        } label: {
          Text("v\(version == 0 ? versions.last?.seq ?? 1 : version)")
        }.menuStyle(.borderlessButton).fixedSize().disabled(dirty || saving)
          .accessibilityLabel("Version").accessibilityIdentifier("artifact-version")
      }
      Spacer(minLength: 0)
      HStack(spacing: 4) {
        Button("Preview") { source = false }
          .buttonStyle(StudioInlineOutlineStyle(selected: !source))
          .accessibilityAddTraits(!source ? .isSelected : [])
        Button {
          source = true
        } label: {
          HStack(spacing: 4) {
            Text("Source")
            if dirty { Circle().frame(width: 5, height: 5).accessibilityLabel("Unsaved changes") }
          }
        }.buttonStyle(StudioInlineOutlineStyle(selected: source))
          .accessibilityAddTraits(source ? .isSelected : [])
      }.fixedSize()
      if dirty {
        Button(saving ? "Saving..." : "Save", action: save).disabled(saving)
          .buttonStyle(StudioInlineOutlineStyle(selected: true))
        Button("Revert", action: revert).disabled(saving)
          .buttonStyle(StudioInlineOutlineStyle())
      }
      HStack(spacing: 0) {
        icon(copied ? "Copied" : "Copy", symbol: copied ? "checkmark" : "doc.on.doc", action: copy)
          .disabled(!available)
        icon("Save to a file", symbol: "arrow.down", action: download).disabled(!available)
        icon("Close preview", symbol: "xmark", action: close)
          .accessibilityIdentifier("close-artifact-preview")
      }.fixedSize()
    }.padding(.vertical, 8).padding(.horizontal, 10)
      .background(PaddockStyle.surface)
      .overlay(alignment: .bottom) {
        Color(nsColor: PaddockStyle.nsColor("borderSubtle")).frame(height: 1)
      }
      .accessibilityIdentifier("native-artifact-header")
  }
  private func icon(_ title: String, symbol: String, action: @escaping () -> Void) -> some View {
    Button(title, systemImage: symbol, action: action)
      .labelStyle(.iconOnly).font(.system(size: 14))
      .frame(width: 26, height: 26).buttonStyle(.plain).help(title)
  }
}
