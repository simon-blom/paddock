import Foundation

/// Shared section order and editor bounds; never a model-dependent slider.
struct StudioSettingsLayout: Decodable {
  struct Section: Decodable, Identifiable {
    let id: String
    let title: String
  }
  struct ReplyLimit: Decodable {
    let maximum: Int
  }
  struct ToolStop: Decodable, Identifiable {
    let value: Int
    let label: String
    var id: Int { value }
  }
  let sections: [Section]
  let replyLimit: ReplyLimit
  let toolStops: [ToolStop]
}
