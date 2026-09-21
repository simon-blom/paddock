import Foundation
import PaddockClient

/// Native presentation of the web ConnectorsPanel's catalog policy.
struct ConnectorCatalog {
  enum Sort: String, CaseIterable { case rank, name, tier, tools, stars, status }
  var sort = Sort.rank
  var ascending = true
  var reachableOnly = false
  var tiers = Set<String>()
  mutating func order(by value: Sort) {
    if sort == value {
      ascending.toggle()
    } else {
      sort = value
      ascending = value != .tools && value != .stars
    }
  }
  func rows(_ hits: [ConnectorHit]) -> [ConnectorHit] {
    let filtered = hits.filter {
      (!reachableOnly || $0.liveness != "dead") && (tiers.isEmpty || tiers.contains(Self.tier($0)))
    }
    guard sort != .rank else { return filtered }
    return filtered.sorted { a, b in
      let comparison: ComparisonResult
      switch sort {
      case .name:
        comparison = (a.name.isEmpty ? a.domain : a.name).localizedStandardCompare(
          b.name.isEmpty ? b.domain : b.name)
      case .tier:
        comparison = compare(Self.tierOrder(a.authorityTier), Self.tierOrder(b.authorityTier))
      case .tools: comparison = compare(a.toolCount ?? 0, b.toolCount ?? 0)
      case .stars: comparison = compare(a.githubStars ?? 0, b.githubStars ?? 0)
      case .status: comparison = compare(Self.statusOrder(a.liveness), Self.statusOrder(b.liveness))
      case .rank: comparison = .orderedSame
      }
      if comparison == .orderedSame { return a.key < b.key }
      return ascending ? comparison == .orderedAscending : comparison == .orderedDescending
    }
  }
  private func compare<T: Comparable>(_ a: T, _ b: T) -> ComparisonResult {
    a == b ? .orderedSame : a < b ? .orderedAscending : .orderedDescending
  }
  static func tier(_ hit: ConnectorHit) -> String {
    hit.authorityTier == "S" ? "First-party" : hit.authorityTier == "A" ? "Trusted" : "Community"
  }
  private static func tierOrder(_ value: String) -> Int {
    ["S": 0, "A": 1, "B": 2, "C": 3, "D": 4][value] ?? 9
  }
  private static func statusOrder(_ value: String) -> Int {
    ["ok-tools": 0, "ok": 1, "auth-required": 2, "dead": 3][value] ?? 9
  }
  static func status(_ hit: ConnectorHit) -> (symbol: String, label: String) {
    switch hit.liveness {
    case "ok", "ok-tools": ("checkmark.circle", "Reachable at last check")
    case "auth-required": ("lock", "Reachable - needs credentials")
    case "dead": ("xmark.circle", "Unreachable at last check")
    default: ("questionmark.circle", "Reachability unknown")
    }
  }
  static func endpoint(_ hit: ConnectorHit) -> String? {
    hit.connection?.recommendedURL ?? hit.remoteEndpoints.first?.url
  }
  static func added(_ hit: ConnectorHit, rows: [NativeConnector]) -> Bool {
    rows.contains { row in
      row.registryKey == hit.key || row.url == endpoint(hit)
        || hit.remoteEndpoints.contains(where: { $0.url == row.url })
    }
  }
  static func slug(_ hit: ConnectorHit) -> String {
    let generic: Set<String> = ["mcp", "api", "www", "app", "gateway", "server"]
    let tail = String(hit.name.split(separator: "/").last ?? "").lowercased()
    let domain =
      hit.domain.split(separator: ".").first { !generic.contains($0.lowercased()) }.map(String.init)
      ?? "connector"
    let raw = tail.isEmpty || generic.contains(tail) ? domain : tail
    let slug = raw.replacingOccurrences(of: "[^a-z0-9_-]+", with: "-", options: .regularExpression)
      .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
    return String((slug.isEmpty ? "connector" : slug).prefix(64))
  }
}
