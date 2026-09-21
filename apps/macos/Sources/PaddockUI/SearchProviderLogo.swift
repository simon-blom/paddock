import SwiftUI

/// SearchLogo.vue, not VendorLogo.vue: search engines keep their brand colours.
/// The SVGs are the web Studio's bundled artwork, cached by ProviderArtwork.
enum SearchProviderArtwork: String, CaseIterable {
  case exa, tavily, firecrawl, brave, perplexity

  var label: String {
    switch self {
    case .exa: "Exa"
    case .tavily: "Tavily"
    case .firecrawl: "Firecrawl"
    case .brave: "Brave"
    case .perplexity: "Perplexity"
    }
  }

  var aspectRatio: CGFloat { self == .firecrawl ? 200.0 / 284.0 : 1 }

  // nil preserves Tavily's original multicolour badge. The other paths are
  // alpha templates; tint them without modifying the shared model-maker image.
  func rgb(dark: Bool) -> UInt32? {
    switch self {
    case .exa: dark ? 0x6f88ff : 0x1f40ed
    case .tavily: nil
    case .firecrawl: 0xfa5d19
    case .brave: 0xfb542b
    case .perplexity: 0x1fb8cd
    }
  }
}

struct SearchProviderLogo: View {
  let provider: SearchProviderArtwork
  var size: CGFloat = 16
  @Environment(\.colorScheme) private var colorScheme

  var body: some View {
    Group {
      if let image = ProviderArtwork.image(for: provider.label) {
        if let rgb = provider.rgb(dark: colorScheme == .dark) {
          Image(nsImage: image).renderingMode(.template).resizable().scaledToFit()
            .foregroundStyle(
              Color(
                .sRGB, red: Double((rgb >> 16) & 255) / 255,
                green: Double((rgb >> 8) & 255) / 255, blue: Double(rgb & 255) / 255, opacity: 1))
        } else {
          Image(nsImage: image).renderingMode(.original).resizable().scaledToFit()
        }
      } else {
        Image(systemName: "globe").resizable().scaledToFit()
      }
    }.frame(width: size * provider.aspectRatio, height: size)
      .accessibilityLabel(provider.label)
      .accessibilityIdentifier("search-provider-logo-\(provider.rawValue)")
  }
}

enum SearchCallIndicator: Equatable {
  case progress, error
  case provider(SearchProviderArtwork)
  case globe

  init(status: String, provider: String) {
    if status == "in_progress" || status == "searching" {
      self = .progress
    } else if status == "failed" {
      self = .error
    } else if let mark = SearchProviderArtwork(rawValue: provider) {
      self = .provider(mark)
    } else {
      self = .globe
    }
  }
}
