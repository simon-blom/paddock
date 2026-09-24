import Foundation
import Testing

@Suite("Concise settings copy")
struct SettingsCopyTests {
  private var root: URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent()
      .deletingLastPathComponent().deletingLastPathComponent().appending(path: "Sources/PaddockUI")
  }

  // Source-copy guard complements the offscreen form-rendering and operation
  // tests. SwiftUI does not publish a usable AX tree in the headless test host.
  @Test func removedBoilerplateDoesNotReturnInOtherSurfaces() throws {
    let forbidden = [
      "\"Saved settings\"", "Saved settings are active.", "Saved keys stay private in Rust",
      "Saved settings. Use Edit", "The runner keeps the key in its private configuration",
      "Hardware, model storage, and app details.", "Model files, verification and recovery.",
      "not model-server configuration.",
      "Closing the window keeps models available.", "Or use the microphone for live transcription",
      "The composer supports dictation, clip recording and live transcription.",
      "Saved key retained", "Leave blank to keep the saved key", "Leave blank to keep current key",
      "Same runner log as web Manager", "logs can still contain paths and operational details",
      "Public servers from the open MCP ecosystem",
      "Send or discard the current draft before opening another conversation.",
      "Credential stored in Keychain", "Credential stored by the shared manager",
      "Your key stays in Keychain.",
      "Add a compatible API endpoint, then choose the models you want in Studio.",
      "Not requested", "Install Paddock in Applications before enabling launch at login.",
      "Allowed by macOS", "Status unavailable",
    ]
    for file in try FileManager.default.contentsOfDirectory(
      at: root, includingPropertiesForKeys: nil)
    where file.pathExtension == "swift" {
      let source = try String(contentsOf: file, encoding: .utf8)
      for phrase in forbidden {
        #expect(!source.contains(phrase), "Redundant copy in \(file.lastPathComponent): \(phrase)")
      }
    }
  }

  @Test func downloadsDoesNotDuplicateCatalogNavigation() throws {
    let source = try String(
      contentsOf: root.appending(path: "DownloadsView.swift"), encoding: .utf8)
    #expect(!source.contains("Browse models"))
    #expect(!source.contains("onBrowse"))
  }

  @Test func actionableWarningsAndCredentialPlaceholdersRemain() throws {
    let required: [String: [String]] = [
      "EndpointSettingsView.swift": [
        "Unsaved changes", "Saved changes awaiting restart",
        "Running settings could not be verified.", "? \"******\"",
        "if let validation = editor.validation", "if let error = editor.error",
      ],
      "EndpointKVOffload.swift": [
        "unencrypted on disk", "if (editor.kvOffloadValue?.nvmeGb ?? 0) > 0",
      ],
      "EndpointMCPSection.swift": ["without per-call approval", "Unlock credentials"],
      "EndpointLogsView.swift": ["if let error = model.error", "textSelection(.enabled)"],
      "WebSearchSettingsView.swift": [
        "if let validation = editor.validation", "if let error = editor.error",
        "editor.keepsSavedKey ? \"******\"",
      ],
      "ConnectionReviewView.swift": [
        "? \"******\"", "A key is saved. Enter a new key to replace it.",
      ],
      "ConnectorReviewView.swift": [
        "Changing its URL removes that sign-in.", "without per-call approval",
      ],
    ]
    for (file, phrases) in required {
      let source = try String(contentsOf: root.appending(path: file), encoding: .utf8)
      for phrase in phrases {
        #expect(source.contains(phrase), "Keep actionable copy in \(file): \(phrase)")
      }
    }
  }

  @Test func newModelFooterDoesNotPretendToEditASavedConfiguration() throws {
    let source = try String(
      contentsOf: root.appending(path: "EndpointSettingsView.swift"), encoding: .utf8)
    let start = try #require(source.range(of: "if !editor.isCreating {"))
    let end = try #require(source.range(of: "Spacer()", range: start.upperBound..<source.endIndex))
    let editOnly = String(source[start.upperBound..<end.lowerBound])
    #expect(
      editOnly.contains("Button(\"Discard changes\")")
        && editOnly.contains("Text(\"Unsaved changes\")"))
    #expect(!source[..<start.lowerBound].contains("Discard changes"))
    #expect(!source[end.upperBound...].contains("Unsaved changes"))
    #expect(
      source.contains(
        ".accessibilityIdentifier(editor.isCreating ? \"start-model-confirm\" : \"endpoint-save\")")
    )
  }
}
