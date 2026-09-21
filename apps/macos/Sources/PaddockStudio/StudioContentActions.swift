import AppKit
import Foundation
import PaddockClient

extension StudioWorkspace {
  public func showArtifact(_ id: String) async {
    guard let conversation = state?.conversation?.id,
      state?.nativeArtifacts?.contains(where: { $0.id == id }) == true
    else { return }
    // Traverse and artifacts share the right-hand slot. Show must work when
    // a graph is already open, without disturbing a left-hand PDF/Word pane.
    do {
      _ = try await command(
        "artifactsPane", ["open": .bool(true), "conversationId": .string(conversation)])
    } catch {
      self.error = error.localizedDescription
      return
    }
    guard state?.conversation?.id == conversation else { return }
    revealArtifact(id)
  }
  /// A preview's close button owns that artifact, never the whole right-hand
  /// slot. This is synchronous presentation state: a stream snapshot cannot
  /// reopen a dismissed preview or make closing it wait on the runtime.
  public func dismissArtifact(_ id: String) {
    guard let artifact = state?.nativeArtifacts?.first(where: { $0.id == id }) else { return }
    dismissedArtifactIDs.insert(id)
    let remaining = presentedArtifacts
    if artifactPicks[artifact.model] == id {
      artifactPicks[artifact.model] = remaining.first { $0.model == artifact.model }?.id
    }
    if selectedArtifactId == id { selectedArtifactId = remaining.first?.id }
  }
  func revealArtifact(_ id: String) {
    guard let artifact = state?.nativeArtifacts?.first(where: { $0.id == id }) else { return }
    dismissedArtifactIDs.remove(id)
    artifactPicks[artifact.model] = id
    selectedArtifactId = id
  }
  public func saveOriginal(_ id: String, name: String) async {
    do {
      let file = try await downloadOriginal(id)
      defer { try? FileManager.default.removeItem(at: file) }
      guard let destination = await saveDestination(name) else { return }
      try await Task.detached {
        try Data(contentsOf: file, options: .mappedIfSafe).write(to: destination, options: .atomic)
      }.value
    } catch { self.error = error.localizedDescription }
  }
  public func saveText(_ text: String, name: String) async {
    guard let destination = await saveDestination(name) else { return }
    do {
      try await Task.detached { try Data(text.utf8).write(to: destination, options: .atomic) }.value
    } catch { self.error = error.localizedDescription }
  }
  private func saveDestination(_ name: String) async -> URL? {
    guard let window = presentationWindow ?? viewerWindow else { return nil }
    let panel = NSSavePanel()
    panel.nameFieldStringValue = URL(fileURLWithPath: name).lastPathComponent
    guard await panel.beginSheetModal(for: window) == .OK else { return nil }
    return panel.url
  }
  public struct ArtifactContent: Decodable, Sendable {
    public struct Version: Decodable, Sendable, Identifiable {
      public let seq: Int
      public let op: String
      public let bytes: Int
      public var id: Int { seq }
    }
    public let body: String
    public let versions: [Version]
    public let revision: Int
  }
  public func artifactContent(_ id: String, version: Int = 0) async throws -> ArtifactContent {
    try validateArtifact(id)
    let conversation = state?.conversation?.id
    var request = try localRequest("api/artifacts/\(id)/content")
    if version > 0 {
      var url = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
      url.queryItems = [.init(name: "version", value: String(version))]
      request.url = url.url
    }
    let metaRequest = try localRequest("api/artifacts/\(id)")
    async let response = contentResponse(request)
    async let meta = contentData(metaRequest)
    struct Metadata: Decodable { let versions: [ArtifactContent.Version]? }
    let (body, headers) = try await response
    guard let revision = Int(headers.value(forHTTPHeaderField: "x-artifact-version") ?? ""),
      revision > 0
    else {
      throw ManagerError.core("The artifact response has no revision. Reload it before editing.")
    }
    let result = try await ArtifactContent(
      body: String(decoding: body, as: UTF8.self),
      versions: JSONDecoder().decode(Metadata.self, from: meta).versions ?? [], revision: revision)
    guard state?.conversation?.id == conversation else {
      throw ManagerError.core("The conversation changed")
    }
    return result
  }
  private func validateArtifact(_ id: String) throws {
    guard id.range(of: "^art_[0-9a-f]{12}$", options: .regularExpression) != nil,
      state?.nativeArtifacts?.contains(where: { $0.id == id }) == true
    else { throw ManagerError.core("This artifact is no longer in the conversation") }
  }
  public func saveArtifact(_ id: String) async throws {
    try validateArtifact(id)
    guard let draft = artifactDrafts[id] else { return }
    guard draft.text.utf8.count <= 4 * 1024 * 1024 else {
      throw ManagerError.core("Artifact edits are limited to 4 MiB. Your draft is retained.")
    }
    let conversation = state?.conversation?.id
    let latest = try await artifactContent(id)
    guard latest.body == draft.saved else {
      throw ManagerError.core(
        "The artifact changed while you edited it. Your draft is retained; review the latest version before saving."
      )
    }
    guard state?.conversation?.id == conversation else {
      throw ManagerError.core("The conversation changed")
    }
    var request = try localRequest("api/artifacts/\(id)/content")
    request.httpMethod = "PUT"
    request.setValue("text/plain; charset=utf-8", forHTTPHeaderField: "Content-Type")
    request.setValue("\"\(latest.revision)\"", forHTTPHeaderField: "If-Match")
    request.httpBody = Data(draft.text.utf8)
    _ = try await contentData(request)
    // Typing while the save is in flight cannot discard the newer edit.
    if let current = artifactDrafts[id], current.text != draft.text {
      var updated = ArtifactDraft(saved: draft.text)
      updated.text = current.text
      artifactDrafts[id] = updated
    } else {
      artifactDrafts[id] = nil
    }
    await perform("refresh")
  }
}
