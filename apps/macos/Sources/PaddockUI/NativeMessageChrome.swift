import AppKit
import PaddockStudio
import SwiftUI

/// Identity and statistics follow Web Studio's message-presentation contract.
/// Native core owns the measured phases and persisted run provenance.
struct NativeMessageHeader: View {
  let message: StudioState.NativeTranscript.Message
  var inLane = false
  var body: some View {
    let name = message.chrome?.modelName ?? message.model
    if !name.isEmpty {
      HStack(spacing: 6) {
        if let vendor = message.chrome?.vendor, !vendor.isEmpty {
          NativeVendorMark(vendor: vendor, size: 14)
            .accessibilityIdentifier("native-message-provider-\(message.id)")
        }
        Text(verbatim: name).font(.system(size: 12, weight: .semibold))
          .lineLimit(1).truncationMode(.tail).textSelection(.enabled)
        if !inLane, let spec = message.chrome?.spec, !spec.isEmpty {
          Text(verbatim: spec).font(.system(size: 10, weight: .medium))
            .padding(.horizontal, 5).padding(.vertical, 2)
            .background(
              PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small)
            )
            .help("Speculation used for this response")
        }
      }
      .foregroundStyle(.secondary).help(name == message.model ? name : "\(name)\n\(message.model)")
      .accessibilityIdentifier("native-message-model-\(message.id)")
    }
  }
}

/// Same assets/letter fallback as VendorLogo.vue, using the native app's
/// intentionally monochrome ink. Do not tint images or Markdown globally.
struct NativeVendorMark: View {
  let vendor: String
  var size: CGFloat = 14
  var body: some View {
    Group {
      if let mark = ProviderArtwork.image(for: vendor) {
        Image(nsImage: mark)
          .renderingMode(ProviderArtwork.usesTemplate(for: vendor) ? .template : .original)
          .resizable().scaledToFit().saturation(0)
      } else {
        Text(String(vendor.prefix(1))).font(.system(size: size * 0.65, weight: .semibold))
          .frame(maxWidth: .infinity, maxHeight: .infinity)
          .background(PaddockStyle.elevated, in: RoundedRectangle(cornerRadius: 3))
      }
    }.frame(width: vendor == "IBM" ? size * 58 / 23 : size, height: size)
      .fixedSize().accessibilityLabel(vendor)
  }
}

struct NativeMessageFooter: View {
  let message: StudioState.NativeTranscript.Message
  var workspace: StudioWorkspace?
  var target: StudioMessageTarget?
  var showsBranches = true
  @Environment(\.transcriptDisclosure) private var readerDisclosure
  @State private var copied = false
  @State private var details = false
  var body: some View {
    if !message.streaming {
      VStack(alignment: .leading, spacing: 10) {
        // MessageBubble.vue: 28px actions, 2px action gaps and a 6px usage
        // margin. Keep controls alongside wrapping metrics, not on a new row.
        HStack(spacing: 8) {
          actions
          metrics
        }
        .frame(maxWidth: .infinity, alignment: message.role == "user" ? .trailing : .leading)
        if details, let chrome = message.chrome { NativeRunDetails(chrome: chrome) }
      }
      .frame(maxWidth: .infinity, alignment: message.role == "user" ? .trailing : .leading)
      .accessibilityIdentifier("native-message-footer-\(message.id)")
      .task(id: copied) {
        guard copied else { return }
        do { try await Task.sleep(for: .seconds(1.5)) } catch { return }
        copied = false
      }
    }
  }
  private var actions: some View {
    HStack(spacing: 2) {
      if !message.text.isEmpty {
        Button(copied ? "Copied" : "Copy message", systemImage: copied ? "checkmark" : "doc.on.doc")
        {
          NSPasteboard.general.clearContents()
          NSPasteboard.general.setString(message.text, forType: .string)
          copied = true
        }.help(copied ? "Copied" : "Copy message")
          .accessibilityIdentifier("native-message-copy-\(message.id)")
      }
      if let workspace, let target, let controls = message.actions {
        if controls.edit {
          Button("Edit and ask again", systemImage: "pencil") { workspace.beginMessageEdit(target) }
            .disabled(!workspace.ready || workspace.busy || workspace.hasMessageEdit)
            .help("Edit and ask again - keeps the original branch")
            .accessibilityIdentifier("native-message-edit-\(message.id)")
        }
        if controls.retry && message.error.isEmpty {
          Button("Retry answer", systemImage: "arrow.clockwise") {
            Task { await workspace.messageAction("retry", target: target) }
          }.disabled(!workspace.ready || workspace.busy || workspace.hasMessageEdit)
            .help("Retry with the selected model - keeps this answer")
            .accessibilityIdentifier("native-message-retry-\(message.id)")
        }
        if showsBranches, let branch = controls.branch {
          HStack(spacing: 2) {
            branchButton(
              "Previous branch", icon: "chevron.left", id: branch.previous, workspace: workspace,
              target: target)
            Text("\(branch.index) / \(branch.count)").font(.system(size: 11)).monospacedDigit()
              .foregroundStyle(.secondary).accessibilityLabel(
                "Branch \(branch.index) of \(branch.count)")
            branchButton(
              "Next branch", icon: "chevron.right", id: branch.next, workspace: workspace,
              target: target)
          }.accessibilityIdentifier("native-message-branches-\(message.id)")
        }
      }
      if let speech = message.speech, let workspace, let target {
        Menu {
          ForEach(["txt", "json", "srt", "vtt"], id: \.self) { format in
            Button("Export \(format.uppercased())") {
              Task { await workspace.exportSpeech(target, format: format) }
            }.disabled(["srt", "vtt"].contains(format) && !speech.subtitleExport)
          }
        } label: {
          Image(systemName: "square.and.arrow.down")
        }
        .menuStyle(.borderlessButton).fixedSize().help("Export transcript")
      }
      if let chrome = message.chrome, !chrome.sections.isEmpty {
        Button(details ? "Hide run details" : "Run details", systemImage: "slider.horizontal.3") {
          readerDisclosure?.reveal()
          details.toggle()
        }
        .help(details ? "Hide run details" : "Run details")
        .background(
          details ? PaddockStyle.elevated : .clear,
          in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.small)
        )
        .accessibilityIdentifier("native-message-details-\(message.id)")
      }
    }
    .labelStyle(.iconOnly).buttonStyle(NativeMessageActionStyle())
    .fixedSize()
  }
  private func branchButton(
    _ label: String, icon: String, id: String?, workspace: StudioWorkspace,
    target: StudioMessageTarget
  ) -> some View {
    Button(label, systemImage: icon) {
      guard let id else { return }
      Task { await workspace.messageAction("branch", target: target, branch: id) }
    }.disabled(id == nil || !workspace.ready || workspace.busy || workspace.hasMessageEdit)
      .help(label).accessibilityIdentifier(
        "native-message-\(icon == "chevron.left" ? "previous" : "next")-\(message.id)")
  }
  @ViewBuilder private var metrics: some View {
    if let chrome = message.chrome, !chrome.footer.isEmpty {
      Text(verbatim: chrome.footer).font(.system(size: 12, design: .monospaced))
        .fixedSize(horizontal: false, vertical: true)
        .foregroundStyle(.secondary).textSelection(.enabled)
        .help(chrome.footerHint)
        .accessibilityIdentifier("native-message-metrics-\(message.id)")
    }
  }
}

/// RunDetails.vue expands inline beneath the footer; no separate popover,
/// fixed width, or nested transcript scroller.
struct NativeRunDetails: View {
  let chrome: StudioState.NativeTranscript.Message.Chrome
  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      ForEach(chrome.sections) { section in
        VStack(alignment: .leading, spacing: 5) {
          Text(verbatim: section.title.uppercased())
            .font(.system(size: 10, weight: .bold)).tracking(0.5).foregroundStyle(.secondary)
          Grid(alignment: .topLeading, horizontalSpacing: 12, verticalSpacing: 3) {
            ForEach(section.rows) { row in
              GridRow(alignment: .top) {
                Text(verbatim: row.label).foregroundStyle(.secondary)
                  .frame(width: 96, alignment: .leading)
                Text(verbatim: row.value).font(.system(size: 12, design: .monospaced))
                  .fixedSize(horizontal: false, vertical: true)
                  .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
              }.font(.system(size: 12))
            }
          }
          if section.id == "provenance", !chrome.promptText.isEmpty {
            DisclosureGroup("Prompt text") {
              PaddockScrollView {
                Text(verbatim: chrome.promptText).font(.system(size: 12, design: .monospaced))
                  .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                  .frame(maxWidth: .infinity, alignment: .leading).padding(.vertical, 8).padding(
                    .horizontal, 10)
              }.frame(maxHeight: 220)
                .background(
                  PaddockStyle.sidebar,
                  in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
                )
                .padding(.top, 6)
            }.font(.system(size: 12)).tint(.secondary).padding(.top, 2)
          }
        }
      }
    }.padding(.vertical, 12).padding(.horizontal, 14)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.card).strokeBorder(PaddockStyle.border)
          .allowsHitTesting(false)
      )
      .accessibilityIdentifier("native-run-details")
  }
}

private struct NativeMessageActionStyle: ButtonStyle {
  @State private var hovered = false
  func makeBody(configuration: Configuration) -> some View {
    configuration.label.font(.system(size: 15)).foregroundStyle(.secondary)
      .frame(width: 28, height: 28)
      .background(
        hovered || configuration.isPressed ? PaddockStyle.surface : .clear,
        in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
      )
      .contentShape(Rectangle()).onHover { hovered = $0 }
  }
}
