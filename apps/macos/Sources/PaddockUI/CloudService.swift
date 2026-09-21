import Foundation
import PaddockClient

/// Same four browse destinations as CloudPanel.vue. Classify known services by
/// their address, not API format: an Anthropic-compatible gateway is still Custom.
enum CloudService: String, CaseIterable, Identifiable {
  case openrouter = "OpenRouter"
  case openai = "OpenAI"
  case anthropic = "Anthropic"
  case custom = "Custom"
  var id: Self { self }
  var vendor: String? { self == .custom ? nil : rawValue }
  var base: String {
    switch self {
    case .openrouter: ConnectionDraft.openRouterBase
    case .openai: "https://api.openai.com/v1"
    case .anthropic: "https://api.anthropic.com/v1"
    case .custom: ""
    }
  }
  var kind: String {
    switch self {
    case .openai: "openai"
    case .anthropic: "anthropic"
    default: "openai-compat"
    }
  }
  static func service(for row: CloudConnection) -> Self {
    allCases.first {
      $0 != .custom && row.baseUrl.trimmingCharacters(in: .init(charactersIn: "/")) == $0.base
    }
      ?? .custom
  }
}
