import PaddockClient

/// Preserve the task that opened setup through selection and catalog browsing.
/// An export's capabilities override the family, including an explicit empty list.
enum ModelStartPurpose: String, CaseIterable, Identifiable {
  case all = "All models"
  case speech = "Speech to text"
  var id: Self { self }

  func matches(_ artifact: CatalogArtifact, model: CatalogModel) -> Bool {
    self == .all || (artifact.runtime?.capability ?? model.capability).contains("transcription")
  }

  func weights(_ model: CatalogModel, backend: String?) -> [CatalogArtifact] {
    // Discovery retains unavailable exports and their support notices. Launch
    // eligibility is stricter and remains enforced by StartModelView and Rust.
    model.weights(on: backend).filter { matches($0, model: model) }
  }
}
