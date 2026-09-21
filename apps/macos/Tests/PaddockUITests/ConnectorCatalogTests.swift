import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Web-aligned connector catalog")
struct ConnectorCatalogTests {
  private func hit(_ key: String, tier: String = "S", status: String = "ok", tools: Int = 1) throws
    -> ConnectorHit
  {
    try JSONDecoder().decode(
      ConnectorHit.self,
      from: JSONSerialization.data(withJSONObject: [
        "key": key, "name": "owner/\(key)", "domain": "api.example.com", "description": "Fixture",
        "authorityTier": tier, "liveness": status, "toolCount": tools, "githubStars": 7,
        "remoteEndpoints": [["url": "https://example.com/old", "transport": "streamable-http"]],
        "connection": ["recommendedURL": "https://example.com/recommended", "authRequired": true],
      ]))
  }
  @Test func rankedCatalogFiltersAndSortsLikeWeb() throws {
    let a = try hit("a", tier: "S", tools: 2)
    let b = try hit("b", tier: "A", tools: 50)
    let c = try hit("c", tier: "B", status: "dead", tools: 100)
    var view = ConnectorCatalog()
    #expect(view.rows([b, a, c]).map(\.key) == ["b", "a", "c"])
    view.reachableOnly = true
    view.order(by: .tools)
    #expect(view.rows([a, b, c]).map(\.key) == ["b", "a"])
    view.order(by: .tools)
    #expect(view.rows([a, b, c]).map(\.key) == ["a", "b"])
    view.tiers = ["First-party"]
    #expect(view.rows([a, b, c]).map(\.key) == ["a"])
  }
  @Test func detailsPreferRecommendedEndpointAndUseWebSlug() throws {
    let value = try hit("github")
    #expect(ConnectorCatalog.endpoint(value) == "https://example.com/recommended")
    #expect(ConnectorCatalog.slug(value) == "github")
    #expect(ConnectorCatalog.tier(value) == "First-party")
    #expect(ConnectorCatalog.status(try hit("x", status: "auth-required")).symbol == "lock")
  }
}
