import Foundation
import PaddockStudio
import Testing

@Suite("Native artifact presentation")
struct ArtifactPresentationTests {
  @Test func csvMatchesTheWebOnSharedFixtures() throws {
    struct Fixture: Decodable {
      let name: String
      let source: String
      let rows: [[String]]
    }
    let url = try #require(
      Bundle.module.url(
        forResource: "artifact-csv", withExtension: "json", subdirectory: "Fixtures"))
    for fixture in try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url)) {
      let csv = try ArtifactCSV.parse(fixture.source)
      #expect(csv.rows == fixture.rows, "\(fixture.name)")
      #expect(csv.totalRows == fixture.rows.count)
      #expect(csv.columns == fixture.rows.map(\.count).max() ?? 0)
    }
  }
  @Test func csvBoundsRetainedCellsButReportsTheCompleteFile() throws {
    let row = Array(repeating: "x", count: 120).joined(separator: ",")
    let csv = try ArtifactCSV.parse(Array(repeating: row, count: 10001).joined(separator: "\n"))
    #expect(csv.rows.count == 501 && csv.rows.allSatisfy { $0.count == 100 })
    #expect(csv.totalRows == 10001 && csv.columns == 120)
    let long = try ArtifactCSV.parse("header\n" + String(repeating: "🦊", count: 5000))
    #expect(long.clippedCells && long.rows[1][0].count == 4096)
    #expect(throws: (any Error).self) {
      try ArtifactCSV.parse(String(repeating: "x", count: 4 * 1024 * 1024 + 1))
    }
  }
  @Test func csvCancellationIsObserved() async {
    let task = Task {
      withUnsafeCurrentTask { $0?.cancel() }
      return try ArtifactCSV.parse("a,b")
    }
    await #expect(throws: CancellationError.self) { try await task.value }
  }
  @Test func artifactPanesFollowChatLanesNotCompletionOrder() throws {
    let artifacts = try [
      artifact("b1", model: "right"), artifact("a1", model: "left"), artifact("b2", model: "right"),
      artifact("legacy", model: ""), artifact("gone", model: "deleted"),
    ]
    let groups = ArtifactPresentation.groups(artifacts, laneOrder: ["left", "right", "left"])
    #expect(groups.map(\.model) == ["left", "right", "deleted"])
    #expect(groups[1].items.map(\.id) == ["b1", "b2"])
    for count in 2...4 {
      #expect(!ArtifactPresentation.splits(groups: count, width: Double(count * 340 - 1)))
      #expect(ArtifactPresentation.splits(groups: count, width: Double(count * 340)))
    }
    #expect(!ArtifactPresentation.splits(groups: 1, width: 2000))
    #expect(!ArtifactPresentation.splits(groups: 4, width: 3 * 340))
    #expect(!ArtifactPresentation.splits(groups: 2, width: .infinity))
  }
  @Test func exportUsesTheArtifactKindAndSafeFilename() throws {
    for (kind, ext) in [
      "html": "html", "svg": "svg", "markdown": "md", "mermaid": "mmd", "csv": "csv",
      "graph": "cypher", "text": "txt",
    ] {
      #expect(
        ArtifactPresentation.filename(try artifact("id", kind: kind, title: "../Test page/"))
          == "Test-page.\(ext)")
    }
    #expect(ArtifactPresentation.filename(try artifact("id", title: "...")) == "id.html")
    #expect(ArtifactPresentation.language(try artifact("id", kind: "svg")) == "xml")
    #expect(ArtifactPresentation.fence("a```b", language: "mermaid") == "````mermaid\na```b\n````")
  }
  @Test func cloudMakerIdentityDoesNotBecomeTheRouterLogo() throws {
    let kimi = try artifact("id", model: "cloud:account:moonshotai/kimi-k2.6")
    let identity = ArtifactPresentation.identity(kimi, state: nil)
    #expect(identity.vendor == "Moonshot")
    #expect(!identity.name.hasPrefix("cloud:"))
  }
  private func artifact(
    _ id: String, model: String = "qwen", kind: String = "html", title: String = "Artifact"
  ) throws -> StudioState.Artifact {
    try JSONDecoder().decode(
      StudioState.Artifact.self,
      from: JSONSerialization.data(withJSONObject: [
        "id": id, "model": model, "kind": kind, "title": title, "language": "", "versions": 1,
        "updatedAt": 1,
      ]))
  }
}
