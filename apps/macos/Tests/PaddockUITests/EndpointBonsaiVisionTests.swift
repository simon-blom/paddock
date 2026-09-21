import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

@Suite("Bonsai optional GGUF vision", .serialized) @MainActor
struct EndpointBonsaiVisionTests {
  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Uses the real Rust catalog projection."))
  func missingCompanionAllowsTextButNeverEnablesVision() throws {
    let model = try catalogModel(visionInstalled: false)
    let weights = try #require(model.artifacts.first { $0.id == "ptq1" })
    #expect(LibraryCatalog.canConfigure(model: model, artifact: weights, backend: "metal"))
    let companions = LibraryCatalog.companions(model: model, artifact: weights, backend: "metal")
    #expect(companions.map(\.id) == ["vision"])
    #expect(companions.first?.installed == false)
    let editor = try editor(model)
    #expect(editor.visionArtifact?.id == "vision" && !editor.embeddedVision)
    #expect(editor.validation == nil && !editor.visionServed)
    #expect(try composition(editor.creationRequest())["vision"] as? Bool == false)
    editor.vision = true
    #expect(editor.validation?.contains("Download the vision companion") == true)
    #expect(editor.creationRequest() == nil)
    editor.context = "16384"
    editor.observeCatalog([try catalogModel(visionInstalled: true)])
    #expect(editor.validation == nil && editor.visionServed)
    #expect(editor.context == "16384", "Download completion must not reset the draft")
    #expect(try composition(editor.creationRequest())["vision"] as? Bool == true)
    editor.observeCatalog([model])
    #expect(editor.creationRequest() == nil, "A removed companion must become unavailable again")
  }

  @Test(
    .enabled(
      if: ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"] != nil,
      "Uses the real Rust catalog projection."))
  func downloadedCompanionIsSelectableAndSavedOffRemainsOff() throws {
    let model = try catalogModel(visionInstalled: true)
    let draft = try editor(model)
    #expect(!draft.vision && !draft.visionServed, "Do not overwrite a saved text-only choice")
    draft.vision = true
    #expect(draft.validation == nil && draft.visionServed)
    #expect(try composition(draft.creationRequest())["vision"] as? Bool == true)
    draft.vision = false
    #expect(try composition(draft.creationRequest())["vision"] as? Bool == false)

    let saved = try editor(model, creating: false)
    #expect(!saved.dirty && !saved.compositionChanged)
    saved.vision = true
    #expect(saved.dirty && saved.compositionChanged && saved.validation == nil)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let changes = try #require(
      JSONSerialization.jsonObject(with: encoder.encode(saved.changes)) as? [[String: Any]])
    let change = try #require(changes.first { $0["field"] as? String == "composition" })
    #expect((change["value"] as? [String: Any])?["vision"] as? Bool == true)
    saved.reset()
    #expect(!saved.vision && !saved.dirty)
  }

  @Test func mandatoryCompanionsStillBlockConfiguration() throws {
    let model = try ManagerWire.decode(
      CatalogModel.self,
      from: Data(
        #"{"id":"speech","display":"Speech","capability":["audio"],"installed":true,"total_size":10,"artifacts":[{"id":"weights","kind":"weights","format":"gguf","label":"Weights","installed":true,"total_size":10,"runtime":{"backends":["metal"],"companions":["audio"]}},{"id":"audio","kind":"audio","format":"gguf","label":"Encoder","installed":false,"required":true,"total_size":10,"runtime":{"backends":["metal"]}}]}"#
          .utf8))
    #expect(
      !LibraryCatalog.canConfigure(model: model, artifact: model.artifacts[0], backend: "metal"))
  }

  private func catalogModel(visionInstalled: Bool) throws -> CatalogModel {
    let path = try #require(ProcessInfo.processInfo.environment["PADDOCK_MACOS_CONTRACT_DIR"])
    let data = try Data(contentsOf: URL(fileURLWithPath: path).appending(path: "catalog.json"))
    let root = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let models = try #require(root["models"] as? [[String: Any]])
    var model = try #require(models.first { $0["id"] as? String == "bonsai-2-27b" })
    var artifacts = try #require(model["artifacts"] as? [[String: Any]])
    for index in artifacts.indices {
      artifacts[index]["installed"] =
        artifacts[index]["id"] as? String == "ptq1"
        || (visionInstalled && artifacts[index]["id"] as? String == "vision")
    }
    model["artifacts"] = artifacts
    return try ManagerWire.decode(
      CatalogModel.self, from: JSONSerialization.data(withJSONObject: model))
  }

  private func editor(_ model: CatalogModel, creating: Bool = true) throws -> EndpointEditor {
    let endpoint = try ManagerWire.decode(
      ConfiguredEndpoint.self,
      from: Data(
        #"{"port":11590,"revision":"fixture-revision","running":false,"model":"bonsai-2-27b","artifact":"ptq1","settings":{"host":"127.0.0.1","max_ctx":32768,"max_batch":1,"spec":"off","kv_cache_dtype":"f16","has_api_key":false,"vision":false,"forensics":false,"device":"metal","runtime_options":[]}}"#
          .utf8))
    return EndpointEditor(
      client: BonsaiVisionClient(), endpoint: endpoint, pid: nil, catalog: [model],
      memoryHardware: .init(physicalBytes: 128 << 30, recommendedBytes: 96 << 30),
      isCreating: creating)
  }

  private func composition(_ request: CreateEndpointRequest?) throws -> [String: Any] {
    let request = try #require(request)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let rows = try #require(
      JSONSerialization.jsonObject(with: encoder.encode(request.changes))
        as? [[String: Any]])
    let row = try #require(rows.first { $0["field"] as? String == "composition" })
    return try #require(row["value"] as? [String: Any])
  }
}

private actor BonsaiVisionClient: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
  func submit(_ command: ModelCommand) async throws -> ManagementJob { throw CancellationError() }
}
