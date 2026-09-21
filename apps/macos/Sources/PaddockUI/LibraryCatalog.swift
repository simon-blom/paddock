import Foundation
import PaddockClient

/// Discovery is scoped by backend and task. Test-progress labels do not affect
/// availability; the declared catalog default remains independent of testing.
enum LibraryFormat: String, CaseIterable, Identifiable {
  case all = "All formats"
  case mlx = "MLX"
  case gguf = "GGUF"
  var id: Self { self }

  func matches(_ artifact: CatalogArtifact) -> Bool {
    switch self {
    case .all: true
    case .mlx: artifact.isMLX
    case .gguf: artifact.format.lowercased() == "gguf"
    }
  }
}

enum LibraryOrder: String, CaseIterable, Identifiable {
  case newest, oldest, nameAscending, nameDescending, smallest, largest
  var id: Self { self }
  var title: String {
    switch self {
    case .newest: "Published · newest first"
    case .oldest: "Published · oldest first"
    case .nameAscending: "Name · A-Z"
    case .nameDescending: "Name · Z-A"
    case .smallest: "Download size · smallest first"
    case .largest: "Download size · largest first"
    }
  }
  var shortTitle: String {
    switch self {
    case .newest: "Newest first"
    case .oldest: "Oldest first"
    case .nameAscending: "Name A-Z"
    case .nameDescending: "Name Z-A"
    case .smallest: "Smallest first"
    case .largest: "Largest first"
    }
  }

  func precedes(_ lhs: LibraryEntry, _ rhs: LibraryEntry) -> Bool {
    switch self {
    case .newest, .oldest:
      // ISO day strings compare chronologically after strict calendar validation.
      // Missing/invalid dates go last in both directions; never use a revision.
      let left = lhs.publishedAt
      let right = rhs.publishedAt
      if left != right {
        guard let left else { return false }
        guard let right else { return true }
        return self == .newest ? left > right : left < right
      }
    case .smallest, .largest:
      // A family offers alternatives, not one download of every quant. Compare
      // its smallest matching weights option, excluding optional companions.
      let left = lhs.smallestDownload
      let right = rhs.smallestDownload
      if left != right {
        guard let left else { return false }
        guard let right else { return true }
        return self == .smallest ? left < right : left > right
      }
    case .nameAscending, .nameDescending: break
    }
    let comparison = lhs.model.display.localizedStandardCompare(rhs.model.display)
    if comparison != .orderedSame {
      return self == .nameDescending
        ? comparison == .orderedDescending : comparison == .orderedAscending
    }
    return lhs.id < rhs.id
  }
}

struct LibraryEntry: Identifiable {
  let model: CatalogModel
  let artifacts: [CatalogArtifact]
  let publishedAt: String?
  let smallestDownload: UInt64?

  init(model: CatalogModel, artifacts: [CatalogArtifact]) {
    self.model = model
    self.artifacts = artifacts
    // Parse once per projection, not once per sort comparison or row draw.
    publishedAt = LibraryCatalog.publicationDate(model)
    smallestDownload = artifacts.map(\.totalSize).filter { $0 > 0 }.min()
  }
  var id: String { model.id }
  var installed: Bool { artifacts.contains(where: \.installed) }
  // A format/search that isolates one export opens exactly that export.
  // Otherwise detail retains its normal, installed/default-aware selection.
  var initialArtifact: String? { artifacts.count == 1 ? artifacts.first?.id : nil }
  var capabilities: Set<String> {
    Set(artifacts.flatMap { $0.runtime?.capability ?? model.capability })
  }

  var preferredArtifact: CatalogArtifact? {
    artifacts.first { $0.installed && $0.default == true }
      ?? artifacts.first(where: \.installed)
      ?? artifacts.first { $0.default == true } ?? artifacts.first
  }
}

struct LibrarySelection: Equatable {
  let model: String
  let artifact: String
}

enum LibraryCatalog {
  static func companions(model: CatalogModel, artifact: CatalogArtifact?, backend: String?)
    -> [CatalogArtifact]
  {
    model.artifacts.filter {
      $0.kind != "weights" && (artifact?.runtime?.companions?.contains($0.id) ?? true)
        && ($0.default == true || $0.required == true) && $0.supports(backend: backend)
    }
  }

  static func canConfigure(model: CatalogModel, artifact: CatalogArtifact, backend: String?) -> Bool
  {
    artifact.installed && artifact.supports(backend: backend)
      && companions(model: model, artifact: artifact, backend: backend).allSatisfy {
        $0.required != true || $0.installed
      }
  }

  static func publicationDate(_ model: CatalogModel) -> String? {
    guard let value = model.specs?.publishedAt,
      value.utf8.count == 10,
      value.range(of: #"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"#, options: .regularExpression) != nil
    else { return nil }
    let parts = value.split(separator: "-").compactMap { Int($0) }
    guard parts.count == 3, parts[0] > 0 else { return nil }
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = TimeZone(secondsFromGMT: 0)!
    let components = DateComponents(year: parts[0], month: parts[1], day: parts[2])
    guard let date = calendar.date(from: components),
      calendar.dateComponents([.year, .month, .day], from: date) == components
    else { return nil }
    return value
  }

  static func webURL(_ value: String?) -> URL? {
    guard let value, let url = URL(string: value), url.scheme == "https",
      let host = url.host, !host.isEmpty, url.user == nil, url.password == nil
    else { return nil }
    return url
  }

  /// Filters select an exact matching export, but the detail pane can explicitly
  /// choose another compatible option without hiding the current model row.
  /// Polling does not reset a user's choice just because installation changes.
  static func selection(
    _ current: LibrarySelection?, in entries: [LibraryEntry], backend: String?,
    requireMatchingExport: Bool = false, purpose: ModelStartPurpose = .all
  ) -> LibrarySelection? {
    guard let entry = entries.first(where: { $0.id == current?.model }) ?? entries.first else {
      return nil
    }
    let allowed =
      requireMatchingExport ? entry.artifacts : purpose.weights(entry.model, backend: backend)
    let chosen =
      allowed.first { $0.id == current?.artifact && entry.id == current?.model }
      ?? entry.preferredArtifact
    return chosen.map { LibrarySelection(model: entry.id, artifact: $0.id) }
  }

  static func entries(
    catalog: ModelCatalog, backend: String?, format: LibraryFormat = .all,
    query: String = "", downloadedOnly: Bool = false, order: LibraryOrder = .newest,
    purpose: ModelStartPurpose = .all
  ) -> [LibraryEntry] {
    let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
    return catalog.models.compactMap { model -> LibraryEntry? in
      let modelMatches =
        query.isEmpty
        || [model.display, model.id, model.vendor ?? ""]
          .contains { $0.localizedStandardContains(query) }
      let artifacts = purpose.weights(model, backend: backend).filter { artifact in
        format.matches(artifact) && (!downloadedOnly || artifact.installed)
          && (modelMatches
            || [
              artifact.id, artifact.label, artifact.quant ?? "", artifact.format,
              artifact.source?.repo ?? "",
            ]
            .contains { $0.localizedStandardContains(query) })
      }
      return artifacts.isEmpty ? nil : LibraryEntry(model: model, artifacts: artifacts)
    }.sorted(by: order.precedes)
  }
}

enum ModelFeature: String, CaseIterable, Identifiable {
  case vision, reasoning, tools
  var id: Self { self }
  var title: String {
    switch self {
    case .vision: "Vision"
    case .reasoning: "Thinking"
    case .tools: "Tools"
    }
  }
  var symbol: String {
    switch self {
    case .vision: "eye"
    case .reasoning: "brain"
    case .tools: "wrench.and.screwdriver"
    }
  }
  func description(in capabilities: Set<String>) -> String {
    "\(title): \(capabilities.contains(rawValue) ? "listed" : "not listed")"
  }
}

enum LibraryRecommendation {
  /// The registry's global default is not a Metal performance or memory-fit
  /// election. Only an explicitly qualified default earns "Recommended".
  static func label(for artifact: CatalogArtifact) -> String? {
    guard artifact.default == true else { return nil }
    return artifact.runtime?.qualification == "qualified" ? "Recommended" : "Catalog default"
  }

  static func explanation(
    model: CatalogModel, backend: String?, purpose: ModelStartPurpose = .all
  ) -> String {
    guard
      let artifact = purpose.weights(model, backend: backend).first(where: { $0.default == true })
    else {
      return
        "No default is specified for this backend. Compare the weights options below."
    }
    if artifact.runtime?.qualification == "qualified" {
      return
        "\(artifact.shortFormat) is recommended for \(backend ?? "this backend"). Memory fit is checked when starting the model."
    }
    return
      "The catalog default is \(artifact.shortFormat). Compare the download sizes and capabilities to choose your weights option."
  }
}

extension CatalogArtifact {
  // Safetensors is a container, not an MLX quantization. CUDA snapshots must
  // never appear in the MLX filter merely because they share that container.
  var isMLX: Bool {
    format.lowercased() == "safetensors" && quant?.uppercased().hasPrefix("MLX-") == true
  }
}
