import SwiftUI

/// One flat trigger for every single-choice menu. The popup itself stays native
/// so selection, keyboard navigation, dismissal and VoiceOver remain system-owned.
struct Dropdown<Content: View>: View {
  let title: String
  let value: String
  var fillsWidth = false
  var vendor: String? = nil
  @ViewBuilder var content: Content

  var body: some View {
    Menu {
      content
    } label: {
      HStack(spacing: 8) {
        if let vendor { ModelProviderLogo(vendor: vendor, size: 16) }
        Text(value).lineLimit(1).truncationMode(.tail)
      }
    }
    .menuStyle(.button).menuIndicator(.hidden)
    .buttonStyle(DropdownButtonStyle(fillsWidth: fillsWidth))
    .accessibilityLabel(title).accessibilityValue(value)
  }
}

private struct DropdownButtonStyle: ButtonStyle {
  let fillsWidth: Bool
  @Environment(\.isEnabled) private var enabled

  func makeBody(configuration: Configuration) -> some View {
    HStack(spacing: 10) {
      configuration.label.frame(maxWidth: fillsWidth ? .infinity : nil, alignment: .leading)
      Image(systemName: "chevron.up.chevron.down")
        .font(.system(size: 8, weight: .medium)).foregroundStyle(.secondary)
        .accessibilityHidden(true)
    }
    .font(.system(size: 12)).foregroundStyle(.primary)
    .padding(.horizontal, 10).frame(height: 30)
    .background(
      configuration.isPressed ? PaddockStyle.elevated : PaddockStyle.canvas,
      in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
    )
    .overlay(
      RoundedRectangle(cornerRadius: PaddockStyle.Radius.control).strokeBorder(
        PaddockStyle.border, lineWidth: 1)
    )
    .contentShape(Rectangle()).opacity(enabled ? 1 : 0.4)
  }
}
