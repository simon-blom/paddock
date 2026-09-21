import Foundation

/// Read-only presentation from studio-settings-layout.ts, shared with the web
/// SettingsPanel. No parallel list of settings or model-derived slider ranges.
struct StudioSettingsLayout: Decodable {
  struct Section: Decodable, Identifiable {
    let id: String
    let title: String
  }
  struct ReplyStop: Decodable {
    let value: Int?
    let label: String
    let shortLabel: String
  }
  struct ToolStop: Decodable, Identifiable {
    let value: Int
    let label: String
    var id: Int { value }
  }
  let sections: [Section]
  let replyStops: [ReplyStop]
  let toolStops: [ToolStop]

  func replyIndex(_ saved: String) -> Int {
    guard let limit = Int(saved) else { return replyStops.count - 1 }
    return replyStops.firstIndex { ($0.value ?? Int.max) >= limit } ?? replyStops.count - 1
  }
}
