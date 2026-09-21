import SwiftUI

/// The same bundled VendorLogo artwork used by the catalog and web form.
struct ModelProviderLogo: View {
  let vendor: String?
  var size: CGFloat = 18
  var body: some View {
    Group {
      if let image = ProviderArtwork.image(for: vendor) {
        Image(nsImage: image).resizable()
          .renderingMode(ProviderArtwork.usesTemplate(for: vendor) ? .template : .original)
          .scaledToFit()
      } else {
        Image(systemName: "cube").resizable().scaledToFit()
      }
    }.frame(width: size, height: size).foregroundStyle(.primary)
      .accessibilityHidden(true)
  }
}

/// Native menu labels need bounded NSImage dimensions, not a resized view.
struct ModelProviderMenuLabel: View {
  let title: String
  let vendor: String?

  var body: some View {
    Label {
      Text(title)
    } icon: {
      if let image = ProviderArtwork.menuImage(for: vendor) {
        Image(nsImage: image)
      } else {
        Image(systemName: "cube")
      }
    }
  }
}
