import Foundation
import Observation
import PaddockStudio
import SwiftUI

typealias StudioCommand =
  @MainActor (String, [String: StudioValue]) async throws -> [String: StudioValue]

func studioDecode<T: Decodable>(_ type: T.Type, _ value: StudioValue?) throws -> T {
  guard let value else { throw StudioLibraryError.message("The Studio reply was incomplete.") }
  return try JSONDecoder().decode(type, from: JSONEncoder().encode(value))
}
enum StudioLibraryError: LocalizedError {
  case message(String)
  var errorDescription: String? {
    switch self {
    case .message(let message): message
    }
  }
}

struct StudioPreset: Codable, Identifiable, Equatable {
  var id: String
  var name: String
  var body: String
  var revision: String
}
struct StudioPresetPage: Decodable {
  struct Row: Decodable, Identifiable {
    let id: String
    let name: String
    let preview: String
    let revision: String
  }
  let rows: [Row]
  let page: Int
  let pageSize: Int
  let total: Int
  let matched: Int
}

@MainActor @Observable final class StudioLibraryModel {
  var command: StudioCommand = { _, _ in
    throw StudioLibraryError.message("Studio is not connected.")
  }
  var onManage: (() -> Void)?
  let instructions = StudioInstructionsModel()
  var query = ""
  private(set) var page: StudioPresetPage?
  private(set) var loading = false
  private(set) var saving = false
  var error: String?
  var notice: String?
  var editor: StudioPreset?
  private var original: StudioPreset?
  private var generation = 0
  @ObservationIgnored private var pending: Task<Void, Never>?
  var dirty: Bool { editor != original }
  var hasWork: Bool { dirty || saving }
  var validation: String? {
    guard let editor else { return "Open a preset first." }
    if editor.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      || editor.body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    {
      return "Enter a name and prompt text."
    }
    if editor.name.utf8.count > 512 || editor.body.utf8.count > 128 * 1024 {
      return "Use a name up to 512 bytes and prompt up to 128 KiB."
    }
    return nil
  }
  func load(page requested: Int = 0) async {
    generation += 1
    let epoch = generation
    let search = query
    loading = true
    defer { if epoch == generation { loading = false } }
    do {
      let result = try await command(
        "promptList", ["search": .string(search), "page": .number(Double(requested))])
      guard epoch == generation, !Task.isCancelled else { return }
      page = try studioDecode(StudioPresetPage.self, result["library"])
      error = nil
    } catch { if epoch == generation { self.error = error.localizedDescription } }
  }
  func create(body: String = "") {
    guard !hasWork else {
      error = "Save or discard the current preset first. Your draft is kept."
      return
    }
    original = nil
    editor = StudioPreset(id: UUID().uuidString, name: "", body: body, revision: "")
    error = nil
    notice = nil
  }
  func open(_ id: String) async {
    guard !hasWork else {
      error = "Save or discard the current preset first. Your draft is kept."
      return
    }
    generation += 1
    let epoch = generation
    loading = true
    defer { if epoch == generation { loading = false } }
    do {
      let result = try await command("promptGet", ["id": .string(id)])
      guard epoch == generation, !hasWork, !Task.isCancelled else { return }
      let record = try studioDecode(StudioPreset.self, result["prompt"])
      editor = record
      original = record
      error = nil
      notice = nil
    } catch { if epoch == generation { self.error = error.localizedDescription } }
  }
  func discard() {
    guard !saving else { return }
    generation += 1
    loading = false
    editor = nil
    original = nil
    error = nil
  }
  func save() {
    guard !saving, let record = editor else { return }
    if let validation {
      error = validation
      return
    }
    mutate {
      let reply = try await self.command(
        "promptSave",
        [
          "id": .string(record.id), "name": .string(record.name), "body": .string(record.body),
          "revision": .string(record.revision),
        ])
      let saved = try studioDecode(StudioPreset.self, reply["prompt"])
      self.editor = saved
      self.original = saved
      self.notice = "Preset saved."
    }
  }
  func remove() {
    guard !saving, !dirty, let record = original else { return }
    mutate {
      _ = try await self.command(
        "promptDelete", ["id": .string(record.id), "revision": .string(record.revision)])
      self.editor = nil
      self.original = nil
      self.notice = "Preset deleted."
      await self.load()
    }
  }
  private func mutate(_ work: @escaping @MainActor () async throws -> Void) {
    saving = true
    error = nil
    notice = nil
    pending = Task {
      defer { self.saving = false }
      do { try await work() } catch { self.error = error.localizedDescription }
    }
  }
  func settle() async {
    await pending?.value
    await instructions.settle()
  }
}

private struct StudioLibraryKey: EnvironmentKey {
  static let defaultValue: StudioLibraryModel? = nil
}
extension EnvironmentValues {
  var studioLibrary: StudioLibraryModel? {
    get { self[StudioLibraryKey.self] }
    set { self[StudioLibraryKey.self] = newValue }
  }
}
