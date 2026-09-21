import PaddockStudio
import SwiftUI

/// Uses the shared model projection; native selection lives in the composer.
struct StudioComposerModelPicker: View {
  @Bindable var chat: StudioWorkspace
  var compact = false
  var compare: () -> Void
  private var header: StudioModelHeader? { chat.state?.modelHeader }
  private var title: String {
    if header?.comparing == true { return "\(header?.compareLanes.count ?? 0) models" }
    return header?.current?.label ?? "Choose a model"
  }
  var body: some View {
    Menu {
      ForEach(header?.pickerOptions ?? []) { option in
        Button {
          Task { await chat.perform("models", ["ids": .array([.string(option.value)])]) }
        } label: {
          if option.value == header?.currentModel && header?.comparing != true {
            Label("\(option.label) · \(option.hint)", systemImage: "checkmark")
          } else {
            Text("\(option.label) · \(option.hint)")
          }
        }.disabled(!option.available).help(option.title)
      }
      if header?.pickerOptions.isEmpty != false {
        Text("Start a local model or add a cloud model in Settings")
      }
      Divider()
      Button("Compare models…", action: compare)
    } label: {
      HStack(spacing: 6) {
        if let vendor = header?.current?.vendor, let image = ProviderArtwork.image(for: vendor) {
          Image(nsImage: image).renderingMode(.template).resizable().scaledToFit()
            .frame(width: 14, height: 14)
        } else {
          Image(systemName: header?.comparing == true ? "rectangle.split.2x1" : "cpu")
        }
        Text(title).font(.system(size: 11, weight: .medium)).lineLimit(1).truncationMode(.middle)
        if !compact, let spec = header?.specLabel, !spec.isEmpty, header?.comparing != true {
          Text(spec).font(.system(size: 9)).foregroundStyle(.secondary).lineLimit(1)
        }
        Image(systemName: "chevron.down").font(.system(size: 8))
      }.frame(maxWidth: compact ? 150 : 230)
    }.menuStyle(.button).buttonStyle(ComposerButtonStyle()).menuIndicator(.hidden)
      .fixedSize(horizontal: false, vertical: true)
      .disabled(!chat.ready || chat.busy || chat.hasMessageEdit)
      .accessibilityLabel("Choose model").accessibilityValue(title)
      .accessibilityIdentifier("composer-model")
      .help(
        header?.comparing == true
          ? (header?.compareLanes.map(\.label).joined(separator: " vs. ") ?? title) : title)
  }
}
