import SwiftUI

/// Shared label/control geometry. A narrow window stacks the same controls;
/// labels never push fields offscreen or rely on unrelated fixed field widths.
struct SettingsRow<Control: View>: View {
  let title: String
  var stacked = false
  var compact = false
  @ViewBuilder var control: () -> Control

  var body: some View {
    let layout =
      stacked
      ? AnyLayout(VStackLayout(alignment: .leading, spacing: 10))
      : AnyLayout(HStackLayout(alignment: compact ? .center : .top, spacing: 24))
    layout {
      Text(title).fontWeight(compact ? .regular : .medium)
        .fixedSize(horizontal: false, vertical: true)
        .frame(width: stacked || compact ? nil : 180, alignment: .leading)
        .padding(.top, stacked || compact ? 0 : 7)
      if compact && !stacked { Spacer(minLength: 0) }
      control().frame(maxWidth: compact ? nil : .infinity, alignment: .leading)
    }.frame(minHeight: 24).accessibilityElement(children: .contain)
  }
}

struct SettingsGroup<Content: View>: View {
  let title: String
  @ViewBuilder var content: () -> Content
  var body: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text(title).font(.system(size: 13, weight: .semibold)).accessibilityAddTraits(.isHeader)
      VStack(alignment: .leading, spacing: 18, content: content)
        .padding(18).frame(maxWidth: .infinity, alignment: .leading)
        .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
    }
  }
}
