import AppKit
import PaddockClient
import SwiftUI

enum ExternalClient: String, CaseIterable, Identifiable {
  case responses = "Responses API"
  case openCode = "OpenCode"
  case openCode2 = "OpenCode 2"
  var id: Self { self }
  func configuration(_ setup: LocalClientSetup) -> String {
    let encoder: (Any) -> String = { value in
      guard
        let bytes = try? JSONSerialization.data(
          withJSONObject: value, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
      else { return "" }
      return String(decoding: bytes, as: UTF8.self)
    }
    let models = [setup.model: ["name": setup.model]]
    switch self {
    case .responses:
      let body = encoder(["model": setup.model, "input": "Say hello.", "stream": true])
      return
        "curl -N \(Self.quote(setup.baseUrl + "/responses")) \\\n  -H 'Content-Type: application/json' \\\n\(setup.hasKey ? "  -H \"Authorization: Bearer $PADDOCK_API_KEY\" \\\n" : "")  --data \(Self.quote(body))"
    case .openCode:
      var options = ["baseURL": setup.baseUrl]
      if setup.hasKey { options["apiKey"] = "{env:PADDOCK_API_KEY}" }
      return encoder([
        "$schema": "https://opencode.ai/config.json", "model": "paddock/\(setup.model)",
        "provider": [
          "paddock": [
            "npm": "@ai-sdk/openai-compatible", "name": "Paddock", "options": options,
            "models": models,
          ]
        ],
      ])
    case .openCode2:
      var provider: [String: Any] = [
        "name": "Paddock", "package": "@opencode/ai/providers/openai-compatible",
        "settings": ["baseURL": setup.baseUrl], "models": models,
      ]
      if setup.hasKey { provider["env"] = ["PADDOCK_API_KEY"] }
      return encoder([
        "$schema": "https://opencode.ai/config.json", "model": "paddock/\(setup.model)",
        "providers": ["paddock": provider],
      ])
    }
  }
  static func quote(_ value: String) -> String {
    "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
  }
}

struct ExternalClientsView: View {
  let client: any ManagerLoading
  let runners: [RunnerInfo]
  @State private var selected: String?
  @State private var format: ExternalClient = .openCode
  @State private var setup: LocalClientSetup?
  @State private var error: String?
  @State private var exporting = false
  @State private var confirmExport = false
  @State private var exportedPath: String?
  private var choices: [RunnerInfo] {
    runners.filter { $0.model != nil && $0.status != "unreachable" }
  }
  private var runner: RunnerInfo? { choices.first { $0.id == selected } ?? choices.first }
  var body: some View {
    PaddockScrollView {
      VStack(alignment: .leading, spacing: 24) {
        PageHeading(title: "Client setup") { EmptyView() }
        if let runner {
          SettingsGroup(title: "Local endpoint") {
            SettingsRow(title: "Instance", compact: true) {
              Dropdown(title: "Instance", value: runner.title) {
                ForEach(choices) { r in Button("\(r.title) · \(r.port)") { selected = r.id } }
              }
            }
            if let setup {
              HStack {
                Text(setup.baseUrl).textSelection(.enabled).font(
                  .system(.body, design: .monospaced))
                Spacer()
                Button("Copy URL", systemImage: "doc.on.doc") { copy(setup.baseUrl) }
              }
              Text(setup.model).foregroundStyle(.secondary).textSelection(.enabled)
            }
          }
          if let setup {
            SettingsGroup(title: "Configuration") {
              Dropdown(title: "Client", value: format.rawValue) {
                ForEach(ExternalClient.allCases) { option in
                  Button(option.rawValue) { format = option }
                }
              }
              Text(format.configuration(setup)).font(.system(size: 12, design: .monospaced))
                .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
              HStack {
                Button("Copy configuration", systemImage: "doc.on.doc") {
                  copy(format.configuration(setup))
                }
                if setup.hasKey {
                  Button(exporting ? "Exporting…" : "Export credentials…", systemImage: "key") {
                    confirmExport = true
                  }.disabled(exporting)
                }
              }
              if let exportedPath {
                HStack {
                  Text("source \(ExternalClient.quote(exportedPath))").font(
                    .system(.caption, design: .monospaced)
                  ).textSelection(.enabled)
                  Spacer()
                  Button("Copy command") { copy("source \(ExternalClient.quote(exportedPath))") }
                }
              }
            }
          }
        } else {
          ContentUnavailableView(
            "Start a chat model first", systemImage: "network",
            description: Text("Running local models provide endpoints for other apps."))
        }
        if let error { Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled) }
      }.padding(32).frame(maxWidth: 900).frame(maxWidth: .infinity)
    }.font(.system(size: 13)).buttonStyle(FlatButtonStyle())
      .task(id: runner?.id) {
        setup = nil
        error = nil
        exportedPath = nil
        guard let runner else { return }
        do {
          let value = try await client.inspect(
            .clientInfo(port: runner.port, pid: runner.pid), as: LocalClientSetup.self)
          try Task.checkCancellation()
          setup = value
        } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
      }
      .confirmationDialog(
        "Export this instance’s API key?", isPresented: $confirmExport, titleVisibility: .visible
      ) {
        Button("Choose export location…") { exportKey() }
        Button("Cancel", role: .cancel) {}
      } message: {
        Text(
          "The file contains a plaintext credential and is readable only by your macOS account. Keep it private and do not add it to source control."
        )
      }
  }
  private func copy(_ text: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(text, forType: .string)
  }
  private func exportKey() {
    guard let runner else { return }
    let panel = NSSavePanel()
    panel.nameFieldStringValue = "paddock-\(runner.port)-credentials.sh"
    panel.canCreateDirectories = true
    panel.begin { response in
      guard response == .OK, let url = panel.url else { return }
      exporting = true
      error = nil
      Task {
        defer { exporting = false }
        do {
          _ = try await client.inspect(
            .exportCredentials(port: runner.port, pid: runner.pid, path: url.path),
            as: NativeExportReceipt.self)
          exportedPath = url.path
        } catch { self.error = error.localizedDescription }
      }
    }
  }
}
