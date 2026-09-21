import AppKit
import Foundation
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Start and Edit share full settings", .serialized) @MainActor
struct EndpointCreationTests {
  @Test func untouchedDraftKeepsPersonalDefaultsAndCannotCallEditOrRemove() async throws {
    let client = CreationClient()
    let editor = try makeEditor(client)
    #expect(!editor.dirty && editor.validation == nil)
    let request = try #require(editor.creationRequest())
    #expect(request.port == nil && !request.allowNetwork)
    let values = try fields(request)
    #expect(values["max_batch"] as? Int == 1)
    #expect(values["max_ctx"] as? Int == 4096)
    #expect(values["spec"] as? String == "on")
    editor.save(.defer)
    editor.remove()
    await editor.settle()
    #expect(await client.mutations == 0)
  }

  @Test func completeDraftCrossesTheBoundaryIncludingAdvancedAndTools() throws {
    let editor = try makeEditor()
    editor.context = "8192"
    editor.concurrency = "4"
    editor.speculation = "adaptive"
    editor.forensics = true
    editor.customMemoryBudget = true
    editor.memoryLimit = "24"
    editor.kvOffloadEnabled = true
    editor.kvOffloadRAM = "2"
    editor.kvOffloadDisk = "16"
    editor.runtimeDraft["temp"] = "0.7"
    editor.runtimeDraft["max_tokens"] = "3072"
    editor.automaticPort = false
    editor.newPort = "11599"
    editor.replacementKey = "synthetic-endpoint-key"
    let tools = EndpointCreationTools(
      provider: "exa", key: "synthetic-search-key",
      connectors: [.init(id: "fixture-tools", revision: 2)])
    let request = try #require(editor.creationRequest(tools: tools))
    let values = try fields(request)
    #expect(request.port == 11599)
    #expect(values.count == 11, "All allowlisted fields, exactly once")
    #expect(values["max_batch"] as? Int == 4 && values["max_ctx"] as? Int == 8192)
    #expect(values["vram_budget"] as? Int == 24576)
    #expect(values["forensics"] as? Bool == true)
    #expect(values["spec"] as? String == "adaptive")
    let runtime = try #require(values["runtime"] as? [String: Any])
    #expect(runtime["max_tokens"] as? Int == 3072 && runtime["temp"] as? Double == 0.7)
    let offload = try #require(values["kv_offload"] as? [String: Any])
    #expect(offload["enabled"] as? Bool == true && offload["nvme_gb"] as? Double == 16)
    #expect(request.tools?.connectors.first?.revision == 2)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let wire = String(decoding: try encoder.encode(ModelCommand.create(request)), as: UTF8.self)
    #expect(wire.contains("synthetic-search-key") && wire.contains("allow_network"))
  }

  @Test func invalidPortRuntimeAndNetworkSettingsCannotStart() throws {
    let editor = try makeEditor()
    editor.host = "0.0.0.0"
    #expect(
      editor.creationRequest(networkConfirmed: true) == nil, "Network needs a key AND consent")
    editor.replacementKey = "synthetic-endpoint-key"
    #expect(editor.creationRequest() == nil)
    #expect(editor.creationRequest(networkConfirmed: true) != nil)
    editor.automaticPort = false
    editor.newPort = "80"
    #expect(editor.creationRequest(networkConfirmed: true) == nil)
    editor.automaticPort = true
    editor.runtimeDraft["max_tokens"] = "abc"
    #expect(editor.creationRequest(networkConfirmed: true) == nil)
    editor.runtimeDraft["max_tokens"] = ""
    editor.concurrency = ""
    #expect(editor.creationRequest(networkConfirmed: true) == nil, "Never fall through to 32 slots")
  }

  @Test func changedSelectionMustBePreparedAndFailurePreservesDraft() async throws {
    let client = CreationClient()
    let editor = try makeEditor(client)
    editor.context = "8192"
    editor.memoryLimit = "20"
    editor.runtimeDraft["temp"] = "0.5"
    editor.artifactID = "another"
    #expect(editor.creationRequest() == nil)
    await editor.prepareCreationSelection()
    #expect(editor.needsReload && editor.error != nil && editor.creationRequest() == nil)
    #expect(
      editor.context == "8192" && editor.memoryLimit == "20" && editor.runtimeDraft["temp"] == "0.5"
    )
    #expect(await client.mutations == 0)
  }

  @Test func startSettingsRenderOffscreenInSimpleAdvancedAndBothThemes() throws {
    let editor = try makeEditor()
    for dark in [false, true] {
      for advanced in [false, true] {
        editor.advanced = advanced
        for width: CGFloat in [600, 1020] {
          let view = NSHostingView(
            rootView: EndpointSettingsView(
              editor: editor, canMutate: true,
              onTools: {}, onCreate: { _ in }
            ).padding(28).frame(width: width)
              .environment(\.colorScheme, dark ? .dark : .light))
          view.frame = NSRect(x: 0, y: 0, width: width, height: 1800)
          view.layoutSubtreeIfNeeded()
          #expect(view.fittingSize.height > 300)
          #expect(view.fittingSize.width <= width)
        }
      }
    }
  }

  private func fields(_ request: CreateEndpointRequest) throws -> [String: Any] {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let rows = try #require(
      JSONSerialization.jsonObject(with: encoder.encode(request.changes)) as? [[String: Any]])
    return Dictionary(uniqueKeysWithValues: rows.map { ($0["field"] as! String, $0["value"]!) })
  }
  private func makeEditor(_ client: CreationClient = CreationClient()) throws -> EndpointEditor {
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"qwen","display":"Qwen","capability":["chat","tools","vision","speculative"],"installed":true,"total_size":16000000000,"artifacts":[{"id":"mlx","kind":"weights","format":"safetensors","label":"MLX","installed":true,"total_size":16000000000,"backend_supported":true,"runtime":{"embedded_vision":true,"kv_cache_dtype":"auto","companions":[],"memory":{"max_ctx":32768,"max_batch":16}}}]}]}"#
          .utf8))
    let endpoint = try ManagerWire.decode(
      ConfiguredEndpoint.self,
      from: Data(
        #"{"port":0,"running":false,"model":"qwen","artifact":"mlx","settings":{"host":"127.0.0.1","max_ctx":4096,"max_batch":1,"spec":"on","kv_cache_dtype":"auto","has_api_key":false,"vision":false,"forensics":false,"device":"metal","kv_offload_supported":true,"runtime_options":[{"id":"temp","label":"Temperature","group":"Generation defaults","kind":"number","minimum":0,"maximum":2,"placeholder":"Model default","help":"","capability":"chat","value":null},{"id":"max_tokens","label":"Output tokens","group":"Generation defaults","kind":"integer","minimum":1,"maximum":1048576,"placeholder":"Model default","help":"","capability":"chat","value":null}]}}"#
          .utf8))
    return EndpointEditor(
      client: client, endpoint: endpoint, pid: nil, catalog: catalog.models,
      memoryHardware: .init(physicalBytes: 64 << 30, recommendedBytes: 48 << 30), isCreating: true)
  }
}

private actor CreationClient: ManagerLoading {
  var mutations = 0
  func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
  func prepareEndpoint(model: String, artifact: String) async throws -> ConfiguredEndpoint {
    throw ManagerError.core("Model no longer installed")
  }
  func submit(_ command: ModelCommand) async throws -> ManagementJob {
    mutations += 1
    throw CancellationError()
  }
}
