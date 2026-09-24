import AppKit
import PaddockStudio
import SwiftUI

struct StudioAttachmentChip: View {
  @Bindable var chat: StudioWorkspace
  let attachment: StudioAttachment
  @State private var options = false
  private var current: StudioAttachment {
    chat.attachments.first(where: { $0.id == attachment.id }) ?? attachment
  }
  private var previewable: Bool {
    current.isAudio || current.isPDF || current.name.lowercased().hasSuffix(".docx")
      || current.mime.hasPrefix("image/")
  }
  private var configurable: Bool {
    chat.state?.composer?.imageMode != true
      && (current.supportsPageSelection || current.mime.hasPrefix("image/"))
  }
  private var summary: String {
    if let error = current.error { return error }
    if !current.ready { return current.phase }
    if current.supportsPageSelection {
      return current.pageSummary + (current.isPDF && current.textOnly ? " · Text only" : "")
    }
    if current.mime.hasPrefix("image/") {
      if chat.state?.composer?.imageMode == true {
        return ByteCountFormatter.string(fromByteCount: Int64(current.size), countStyle: .file)
      }
      let label = StudioAttachment.imageDetailOptions.first { $0.value == current.detail }?.title
      return [label, StudioImageEstimate.label(chat.imageEstimates(for: current))]
        .compactMap { $0 }.joined(separator: " · ")
    }
    return ByteCountFormatter.string(fromByteCount: Int64(current.size), countStyle: .file)
  }
  var body: some View {
    HStack(spacing: 8) {
      Button(action: preview) {
        if let bytes = current.thumbnail, let image = NSImage(data: bytes) {
          Image(nsImage: image).resizable().scaledToFill().frame(width: 32, height: 36)
            .clipShape(RoundedRectangle(cornerRadius: PaddockStyle.Radius.control))
        } else {
          Image(
            systemName: current.isAudio
              ? "waveform"
              : current.isPDF
                ? "doc.richtext" : current.mime.hasPrefix("image/") ? "photo" : "doc"
          )
          .font(.system(size: 19)).frame(width: 32, height: 36)
        }
      }.buttonStyle(QuietButtonStyle()).disabled(!current.ready || !previewable)
        .accessibilityLabel("Preview \(current.name)")
      VStack(alignment: .leading, spacing: 4) {
        Button(current.name, action: preview)
          .buttonStyle(QuietButtonStyle()).disabled(!current.ready || !previewable)
          .lineLimit(1).help(current.error ?? "Open \(current.name)")
        if current.ready && configurable {
          Button {
            options = true
          } label: {
            HStack(spacing: 4) {
              Text(summary).lineLimit(1)
              Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
            }.font(.system(size: 10)).foregroundStyle(.secondary)
          }.buttonStyle(QuietButtonStyle()).disabled(chat.busy)
            .accessibilityLabel("\(current.name): \(summary). Attachment options")
            .accessibilityIdentifier("attachment-options-\(current.id)")
        } else {
          Text(summary).font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
        }
      }.frame(width: 178, alignment: .leading)
      Button("Remove \(current.name)", systemImage: "xmark") {
        chat.removeAttachment(current.id)
      }.labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).disabled(chat.busy)
    }.font(.system(size: 12)).padding(9)
      .background(PaddockStyle.canvas, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.card))
      .overlay(
        RoundedRectangle(cornerRadius: PaddockStyle.Radius.card).strokeBorder(PaddockStyle.border)
      )
      // The presenter belongs to the stable attachment, not the conditional
      // summary button whose label/geometry changes with every page edit.
      .popover(isPresented: $options, arrowEdge: .top) {
        StudioAttachmentOptions(
          attachment: Binding(
            get: { current },
            set: { value in
              guard !chat.busy,
                let i = chat.attachments.firstIndex(where: { $0.id == current.id })
              else { return }
              chat.attachments[i] = value
            }),
          canRasterPDF: chat.state?.capabilities.vision == true
            && chat.state?.capabilities.pdfRaster == true,
          capabilities: chat.state?.capabilities,
          onDone: { options = false })
      }
      .onChange(of: chat.busy) { _, busy in if busy { options = false } }
  }
  private func preview() {
    Task { await chat.perform("preview", ["id": .string(current.id)]) }
  }
}
