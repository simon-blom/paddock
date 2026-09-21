import Foundation
import PaddockClient
import Testing

@testable import PaddockUI

/// Source structure guard supplements the light/dark native rendering checks.
/// Local defaults must not replace the web Manager's visible controls.
@Suite("Web Manager settings parity") @MainActor
struct WebManagerSettingsParityTests {
  private var root: URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent()
  }
  private func source(_ name: String) throws -> String {
    try String(contentsOf: root.appending(path: "Sources/PaddockUI/\(name).swift"), encoding: .utf8)
  }

  private func compact(_ text: String) -> String {
    String(text.filter { !$0.isWhitespace })
  }

  @Test func weightsAndWorkloadRemainCardsWithTheWebInformationOrder() throws {
    let native = try source("EndpointModelWorkload")
    let web = try String(
      contentsOf: root.deletingLastPathComponent().deletingLastPathComponent()
        .appending(path: "studio/src/components/manage/ServerForm.vue"), encoding: .utf8)
    let labels = [
      "Quality", "Workload", "Context per conversation", "Conversation memory", "Speculative",
      "Drafter",
    ]
    var previous = native.startIndex
    for label in labels {
      let range = try #require(
        native.range(of: "EndpointFormField(\"\(label)\"", range: previous..<native.endIndex)
          ?? (label == "Context per conversation"
            ? native.range(of: "\"Context per conversation\"", range: previous..<native.endIndex)
            : nil))
      previous = range.upperBound
      #expect(web.contains(label), "Keep the web control: \(label)")
    }
    #expect(native.components(separatedBy: "LazyVGrid(").count - 1 == 2)
    #expect(native.contains("DisplayFormat.bytes(artifact.totalSize)"))
    #expect(native.contains("artifact.quant ?? artifact.format"))
    #expect(native.contains(".isSelected"))
    #expect(!native.contains("Dropdown(title: \"Workload\""))
    #expect(
      !native.contains("if editor.advanced"), "Simple retains memory/speculation/drafter controls")
  }

  @Test func simpleRetainsAllEditableSectionsWithoutDuplicateSummary() throws {
    let native = try source("EndpointSettingsView")
    let simple = try #require(native.range(of: "private var simple: some View"))
    let access = try #require(native.range(of: "private func access(advanced:"))
    let body = String(native[simple.lowerBound..<access.lowerBound])
    let sections = [
      "Model & workload", "EndpointMemorySettings", "EndpointKVOffloadSettings",
      "Document & image intelligence", "File metadata", "System tools", "access(advanced: false)",
    ]
    var previous = body.startIndex
    for section in sections {
      let range = try #require(body.range(of: section, range: previous..<body.endIndex))
      previous = range.upperBound
    }
    #expect(native.contains(".pickerStyle(.segmented)"))
    #expect(!native.contains("EndpointConfigurationSummary("))
    #expect(try source("EndpointEditView").contains("frame(maxWidth: 1080"))
  }

  @Test func instanceCreationStaysInlineAndCatalogIsTheOnlyModelPicker() throws {
    let workspace = try source("WorkspaceView")
    let start = try source("StartModelView")
    let instances = try source("EndpointsView")
    #expect(
      compact(workspace).contains(
        compact(
          "onCreate: { navigation.showModelLibrary(purpose: .all) }, creation: instanceCreation(snapshot)"
        )))
    #expect(!workspace.contains("startSelection = StartSelection()"))
    #expect(!workspace.contains("if let chosen = startSelection {"))
    #expect(instances.contains("creation.accessibilityIdentifier(\"instance-creation\")"))
    #expect(!start.contains("StartModelPicker(") && !start.contains("private var versions"))
    #expect(!start.contains("PaddockScrollView {"), "The Instances page owns scrolling")
    #expect(
      !FileManager.default.fileExists(
        atPath: root.appending(path: "Sources/PaddockUI/StartModelPicker.swift").path))
    #expect(
      !FileManager.default.fileExists(
        atPath: root.appending(path: "Sources/PaddockUI/EndpointConfigurationSummary.swift").path))
  }

  @Test func installedCatalogDefaultIsPreferredWithoutChangingExplicitChoices() throws {
    let base = try endpointFixture()
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"test","display":"Test","capability":["chat"],"installed":true,"total_size":10,"artifacts":[{"id":"first","kind":"weights","format":"gguf","label":"GGUF","installed":true,"total_size":10,"backend_supported":true},{"id":"preferred","kind":"weights","format":"safetensors","label":"MLX","installed":true,"total_size":10,"backend_supported":true,"default":true},{"id":"missing","kind":"weights","format":"gguf","label":"Not downloaded","installed":false,"total_size":10,"backend_supported":true}]}]}"#
          .utf8))
    let snapshot = ManagerSnapshot(
      identity: base.identity, readiness: base.readiness, catalog: catalog, runners: [])
    let unselected = StartModelView(snapshot: snapshot) { _ in false }
    #expect(unselected.initialModel.isEmpty && unselected.initialArtifact.isEmpty)
    #expect(unselected.request == nil)
    let view = StartModelView(snapshot: snapshot, model: "test") { _ in false }
    #expect(view.initialArtifact == "preferred" && view.validation == nil)
    #expect(view.request == nil, "No unprepared simplified request can bypass Edit's schema")
    let explicit = StartModelView(snapshot: snapshot, model: "test", artifact: "first") { _ in false
    }
    #expect(explicit.initialArtifact == "first" && explicit.validation == nil)
    let unavailable = StartModelView(snapshot: snapshot, model: "test", artifact: "missing") { _ in
      false
    }
    #expect(unavailable.request == nil && unavailable.validation != nil)
    let speech = StartModelView(snapshot: snapshot, purpose: .speech) { _ in false }
    #expect(speech.request == nil, "Chat weights cannot be selected as speech defaults")
  }

  @Test func startAndEditUseTheSameSettingsForm() throws {
    let start = try source("StartModelView")
    let edit = try source("EndpointEditView")
    #expect(compact(start).contains("EndpointSettingsView(editor:editor"))
    #expect(compact(edit).contains("EndpointSettingsView(editor:editor"))
    #expect(!start.contains("DisclosureGroup(\"Advanced\""))
    #expect(!start.contains("useDefaults"))
    #expect(start.contains("WebSearchForm(editor: search)"))
    #expect(start.contains("EndpointMCPSection(model: tools"))
    #expect(start.contains("creationRequest(networkConfirmed:"))
  }
}
