import Foundation

/// Actor-owned presentation reuse. Unrelated stream/history updates must not
/// reparse every completed OCR table. This is not durable state and never
/// replaces the authoritative raw extraction. Budget includes raw + display.
struct NativeOCRDisplayCache: Sendable {
  struct Entry: Sendable {
    let raw: String
    let display: String
    let bytes: Int
  }
  private var entries: [String: Entry] = [:]
  private var order: [String] = []
  private(set) var bytes = 0
  mutating func display(id: String, raw: String) -> String {
    if let entry = entries[id], entry.raw == raw { return entry.display }
    if let old = entries.removeValue(forKey: id) { bytes -= old.bytes }
    order.removeAll { $0 == id }
    let display = NativeOCRText.display(raw)
    guard raw != display else { return raw }
    let cost = raw.utf8.count + display.utf8.count
    guard cost <= 8 * 1024 * 1024 else { return display }
    while !order.isEmpty && (bytes + cost > 8 * 1024 * 1024 || entries.count >= 64) {
      if let old = entries.removeValue(forKey: order.removeFirst()) { bytes -= old.bytes }
    }
    entries[id] = Entry(raw: raw, display: display, bytes: cost)
    order.append(id)
    bytes += cost
    return display
  }
}
