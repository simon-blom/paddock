import AppKit
import PaddockClient
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Metal model settings", .serialized) @MainActor
struct EndpointMetalSettingsTests {
  @Test func bonsaiShowsItsNativeF32ContractWithoutOfferingBF16OrADrafter() throws {
    let editor = try editor()
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"qwen","display":"Bonsai 2 27B","capability":["chat","vision","tools","reasoning"],"installed":true,"total_size":8595477990,"artifacts":[{"id":"mlx","kind":"weights","format":"safetensors","label":"Prism ML 2-bit","installed":true,"total_size":8595477990,"backend_supported":true,"runtime":{"checkpoint_dir":true,"embedded_vision":true,"kv_cache_dtype":"f32","capability":["chat","vision","tools","reasoning"],"companions":[]}}]}]}"#
          .utf8))
    editor.catalog = catalog.models
    editor.selectArtifact(try #require(editor.selectedArtifact), newModel: true)
    #expect(editor.isMLX && editor.embeddedVision)
    #expect(editor.kvLabel == "F32 · checkpoint native" && editor.kvChoices == ["", "f32"])
    #expect(editor.kvDtype == "f32" && !editor.canSpeculate && editor.drafters.isEmpty)
  }
  @Test func splashUsesItsBundledDraftAndNativeBF16Labels() throws {
    let editor = try editor()
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"qwen","display":"Qwen 3.8 27B","capability":["chat","vision","speculative"],"installed":true,"total_size":17382689804,"artifacts":[{"id":"mlx","kind":"weights","format":"splash-packed-q4","label":"Splash 4-bit","installed":true,"total_size":17382689804,"backend_supported":true,"runtime":{"checkpoint_dir":true,"embedded_vision":true,"kv_cache_dtype":"auto","companions":[]}}]}]}"#
          .utf8))
    editor.catalog = catalog.models
    #expect(editor.isSplash && !editor.isMLX && editor.embeddedVision)
    #expect(editor.checkpointLabel == "Splash packed Q4")
    #expect(editor.kvLabel == "BF16 · checkpoint native" && editor.kvChoices == ["", "auto"])
    #expect(editor.drafters.isEmpty && editor.speculationSummary == "Bundled DFlash2")
  }
  @Test func clearingWorkloadCannotSilentlySelect32Slots() throws {
    let editor = try editor()
    editor.concurrency = ""
    #expect(editor.validation?.contains("Just me") == true)
    editor.concurrency = "1"
    #expect(editor.validation == nil && editor.effectiveConcurrency == 1)
    editor.reset()
    #expect(
      editor.effectiveConcurrency == 4 && !editor.dirty,
      "Restoring the draft must preserve the user's existing saved workload")
  }
  @Test func offloadBudgetIsValidatedAndUsesTheTypedWebConfig() throws {
    let editor = try editor()
    #expect(!editor.kvOffloadDirty)
    editor.kvOffloadEnabled = true
    #expect(editor.dirty && editor.kvOffloadValidation != nil)
    editor.kvOffloadRAM = "2"
    editor.kvOffloadDisk = "32"
    #expect(editor.kvOffloadValidation == nil)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let json = String(decoding: try encoder.encode(editor.changes), as: UTF8.self)
    #expect(json.contains("kv_offload") && json.contains("ram_gb") && json.contains("nvme_gb"))
    editor.kvOffloadRAM = "nan"
    #expect(editor.kvOffloadValidation != nil)
    editor.kvOffloadRAM = "65"
    #expect(editor.kvOffloadValidation?.contains("physical RAM") == true)
    editor.reset()
    #expect(!editor.kvOffloadEnabled && !editor.kvOffloadDirty && !editor.dirty)
  }
  @Test func offloadControlsRenderInBothNativeThemes() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      let editor = try editor()
      editor.kvOffloadEnabled = true
      editor.kvOffloadRAM = "2"
      editor.kvOffloadDisk = "32"
      let host = NSHostingController(
        rootView: EndpointKVOffloadSettings(editor: editor)
          .padding(24).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
          .background(PaddockStyle.canvas).environment(\.colorScheme, dark ? .dark : .light))
      host.sizingOptions = []
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 620, height: 460),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentViewController = host
      window.setContentSize(NSSize(width: 620, height: 460))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      try await Task.sleep(for: .milliseconds(100))
      host.view.layoutSubtreeIfNeeded()
      #expect(abs(host.view.frame.width - 620) < 1)
      if let root = ProcessInfo.processInfo.environment["PADDOCK_ENDPOINT_SNAPSHOTS"],
        let bitmap = host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds)
      {
        host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
        try bitmap.representation(using: .png, properties: [:])?.write(
          to: URL(fileURLWithPath: root).appending(
            path: "kv-offload-\(dark ? "dark" : "light").png"))
      }
    }
  }
  @Test func memoryLimitIsExplicitValidatedAndNeverInvents32GiB() throws {
    let editor = try editor()
    #expect(!editor.customMemoryBudget && editor.memoryLimit.isEmpty && !editor.dirty)
    editor.setCustomMemoryBudget(true)
    #expect(editor.memoryLimit.isEmpty && editor.dirty && editor.memoryValidation != nil)
    editor.memoryLimit = "nan"
    #expect(editor.memoryValidation != nil)
    editor.memoryLimit = "24.5"
    #expect(editor.memoryBudgetMiB == 25088 && editor.memoryValidation == nil)
    let changes = String(decoding: try JSONEncoder().encode(editor.changes), as: UTF8.self)
    #expect(changes.contains("25088"))
    editor.memoryLimit = "24\(Locale.current.decimalSeparator ?? ".")5"
    #expect(editor.memoryBudgetMiB == 25088)
    editor.memoryLimit = "49"
    #expect(editor.memoryValidation?.contains("48 GiB") == true)
    editor.setCustomMemoryBudget(false)
    #expect(editor.memoryLimit.isEmpty && !editor.dirty && editor.memoryValidation == nil)
    #expect(MetalMemoryHardware.gib(48 << 30) == "48 GiB")
  }
  @Test func memoryControlRendersAutomaticAndCustomWithoutASidebarSqueeze() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      let editor = try editor()
      let controller = NSHostingController(
        rootView: EndpointMemorySettings(editor: editor)
          .padding(24).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
          .background(PaddockStyle.canvas).environment(\.colorScheme, dark ? .dark : .light))
      controller.sizingOptions = []
      let window = NSWindow(
        contentRect: NSRect(x: -12000, y: -12000, width: 580, height: 440),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentViewController = controller
      window.setContentSize(NSSize(width: 580, height: 440))
      window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
      window.orderBack(nil)
      defer { window.close() }
      let host = controller.view
      host.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      for state in ["auto", "custom"] {
        if state == "custom" {
          editor.setCustomMemoryBudget(true)
          editor.memoryLimit = "24"
        }
        try await Task.sleep(for: .milliseconds(100))
        host.layoutSubtreeIfNeeded()
        #expect(abs(host.frame.width - 580) < 1)
        let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        if let directory = ProcessInfo.processInfo.environment["PADDOCK_ENDPOINT_SNAPSHOTS"] {
          try bitmap.representation(using: .png, properties: [:])?.write(
            to: URL(fileURLWithPath: directory)
              .appending(path: "memory-\(state)-\(dark ? "dark" : "light").png"))
        }
      }
    }
  }
  @Test func legacySpeculationAndMissingDimensionsMatchRunnerSemantics() throws {
    for (raw, disabled, expected) in [
      ("auto", false, "adaptive"), ("ladder", false, "on"), ("on", true, "off"),
    ] {
      let editor = try editor(spec: raw, noSpec: disabled)
      #expect(editor.speculation == expected && !editor.dirty)
      editor.context = ""
      editor.concurrency = ""
      #expect(editor.effectiveContext == 4096 && editor.effectiveConcurrency == 32)
      #expect(editor.recommendedContext == 8192 && editor.recommendedConcurrency == 1)
      editor.speculation = expected == "off" ? "on" : "off"
      let data = try JSONEncoder().encode(editor.changes)
      #expect(String(decoding: data, as: UTF8.self).contains("\"spec\""))
    }
  }
  @Test func exactExportControlsMemoryVisionAndDrafterAvailability() throws {
    let editor = try editor()
    #expect(editor.isMLX && editor.embeddedVision && editor.canSpeculate)
    #expect(editor.kvChoices == ["", "auto"])
    #expect(
      editor.electedDrafter?.id == "dflash2" && editor.speculationSummary.contains("DFlash 2"))
    #expect(!editor.weights.contains { $0.id == "cuda" })
    editor.kvDtype = "fp8_e4m3"
    #expect(editor.validation != nil)
    editor.reset()
    editor.concurrency = "17"
    #expect(editor.validation?.contains("16") == true)
    editor.reset()
    editor.context = "65536"
    #expect(editor.validation?.contains("32") == true)
  }
  @Test func choosingWeightsOrAnotherModelPreservesTheUsersWorkload() throws {
    let editor = try editor()
    editor.concurrency = "1"
    let gguf = try #require(editor.weights.first { $0.id == "gguf" })
    editor.selectArtifact(gguf, newModel: true)
    #expect(editor.concurrency == "1" && editor.effectiveConcurrency == 1)
    let changes = try #require(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(editor.changes)) as? [[String: Any]])
    #expect(changes.contains { $0["field"] as? String == "max_batch" && $0["value"] as? Int == 1 })
    editor.concurrency = "4"
    editor.selectArtifact(try #require(editor.weights.first { $0.id == "mlx" }), newModel: true)
    #expect(editor.concurrency == "4", "Explicit agent/server workloads are preserved too")
  }
  @Test func runtimeOptionsValidateAndEncodeOnlyChangedFieldsIncludingNull() throws {
    let editor = try editor()
    #expect(!editor.dirty)
    editor.runtimeDraft["temp"] = "nan"
    #expect(editor.validation?.contains("temperature") == true)
    editor.runtimeDraft["temp"] = "0.3"
    editor.runtimeDraft["top_p"] = ""
    editor.runtimeDraft["no_metrics"] = "true"
    #expect(editor.validation == nil)
    editor.advanced = true
    editor.advanced = false
    let changes = try #require(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(editor.changes)) as? [[String: Any]])
    #expect(changes.count == 1 && changes[0]["field"] as? String == "runtime")
    let value = try #require(changes[0]["value"] as? [String: Any])
    #expect(value["temp"] as? Double == 0.3)
    #expect(value["top_p"] is NSNull)
    #expect(value["no_metrics"] as? Bool == true)
    #expect(value["api_key"] == nil && value["model"] == nil)
    editor.reset()
    #expect(!editor.dirty && editor.runtimeDraft["top_p"] == "0.9")
  }
  @Test func simpleAndAdvancedMountOffscreenWithRealMetalChoices() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      for width: CGFloat in [640, 1000] {
        let editor = try editor()
        let host = NSHostingController(
          rootView: ScrollView {
            EndpointSettingsView(editor: editor, canMutate: true, onTools: {})
              .padding(24).frame(maxWidth: .infinity)
          }.background(PaddockStyle.canvas).environment(\.colorScheme, dark ? .dark : .light))
        let window = NSWindow(
          contentRect: NSRect(x: -12000, y: -12000, width: width, height: 800),
          styleMask: [.borderless], backing: .buffered, defer: false)
        host.sizingOptions = []
        window.isReleasedWhenClosed = false
        window.contentViewController = host
        window.setContentSize(NSSize(width: width, height: 800))
        window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
        window.orderBack(nil)
        defer { window.close() }
        for advanced in [false, true, false] {
          editor.advanced = advanced
          try await Task.sleep(for: .milliseconds(120))
          host.view.layoutSubtreeIfNeeded()
          #expect(abs(host.view.frame.width - width) < 1)
          #expect(!editor.dirty)
          let texts = descendants(host.view).compactMap { $0 as? NSTextField }.map(\.stringValue)
          #expect(!texts.contains("Not available for Metal"))
          if let root = ProcessInfo.processInfo.environment["PADDOCK_ENDPOINT_SNAPSHOTS"],
            let bitmap = host.view.bitmapImageRepForCachingDisplay(in: host.view.bounds)
          {
            host.view.cacheDisplay(in: host.view.bounds, to: bitmap)
            try bitmap.representation(using: .png, properties: [:])?.write(
              to: URL(fileURLWithPath: root).appending(
                path:
                  "metal-\(advanced ? "advanced" : "simple")-\(dark ? "dark" : "light")-\(Int(width)).png"
              ))
          }
        }
      }
    }
  }
  private func descendants(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(descendants)
  }
  private struct NoCore: ManagerLoading {
    func snapshot() async throws -> ManagerSnapshot { throw CancellationError() }
  }
  private func editor(spec: String = "on", noSpec: Bool = false) throws -> EndpointEditor {
    let catalog = try ManagerWire.decode(
      ModelCatalog.self,
      from: Data(
        #"{"schema":3,"models":[{"id":"qwen","display":"Qwen 3.8 27B","capability":["chat","tools","vision","speculative"],"installed":true,"total_size":16000000000,"artifacts":[{"id":"mlx","kind":"weights","format":"safetensors","label":"MLX · 4-bit","installed":true,"total_size":16000000000,"backend_supported":true,"runtime":{"embedded_vision":true,"kv_cache_dtype":"auto","companions":["dflash2"],"default_max_ctx":8192,"default_max_batch":4,"memory":{"max_ctx":32768,"max_batch":16}}},{"id":"gguf","kind":"weights","format":"gguf","label":"GGUF · Q4_K","installed":true,"total_size":17000000000,"backend_supported":true,"runtime":{"capability":["chat","tools"],"kv_cache_dtype":"f16","companions":[]}},{"id":"cuda","kind":"weights","format":"safetensors","label":"FP8","installed":true,"total_size":27000000000,"backend_supported":false},{"id":"dflash2","kind":"drafter","format":"gguf","label":"DFlash 2","installed":true,"total_size":1000000000,"backend_supported":true,"default":true}]}]}"#
          .utf8))
    let settings: [String: Any] = [
      "host": "127.0.0.1", "max_ctx": 8192, "max_batch": 4, "spec": spec, "no_spec": noSpec,
      "kv_offload_supported": true,
      "kv_cache_dtype": "auto", "has_api_key": true, "vision": false, "forensics": false,
      "device": "metal",
      "runtime_options": [
        option("temp", "Temperature", "Generation defaults", "number", 0, 2, nil),
        option("top_p", "Top P", "Generation defaults", "number", 0, 1, 0.9),
        option("no_metrics", "Disable metrics", "Diagnostics", "boolean", 0, 0, nil),
      ],
    ]
    let endpoint = try ManagerWire.decode(
      ConfiguredEndpoint.self,
      from: JSONSerialization.data(withJSONObject: [
        "port": 12345, "model": "qwen", "artifact": "mlx", "revision": "original", "running": false,
        "settings": settings,
      ]))
    return EndpointEditor(
      client: NoCore(), endpoint: endpoint, pid: nil, catalog: catalog.models,
      memoryHardware: .init(physicalBytes: 64 << 30, recommendedBytes: 48 << 30))
  }
  private func option(
    _ id: String, _ label: String, _ group: String, _ kind: String, _ min: Double, _ max: Double,
    _ value: Any?
  ) -> [String: Any] {
    [
      "id": id, "label": label, "group": group, "kind": kind, "minimum": min, "maximum": max,
      "placeholder": "Model default", "help": "Requests can override this default.",
      "capability": "", "value": value ?? NSNull(),
    ]
  }
}
