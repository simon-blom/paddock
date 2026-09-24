import Foundation
import Observation
import PaddockStudio

@MainActor @Observable final class StudioPreferencesModel {
  var command: StudioCommand = { _, _ in
    throw StudioLibraryError.message("Studio is not connected.")
  }
  var reply = ReplyLimitDraft()
  var toolLimit = ""
  var summarize = true
  var mapTiles = ""
  private(set) var layout: StudioSettingsLayout?
  private(set) var mapHost = ""
  private(set) var loaded = false
  private(set) var loading = false
  private(set) var saving = false
  var error: String?
  var notice: String?
  private var original: [String: StudioValue] = [:]
  @ObservationIgnored private var pending: Task<Void, Never>?
  var dirty: Bool { loaded && (try? values()) != original }
  var hasWork: Bool { dirty || saving }
  var validation: String? {
    do {
      _ = try values()
      return nil
    } catch { return error.localizedDescription }
  }
  private func values() throws -> [String: StudioValue] {
    if let validation = reply.validation { throw StudioLibraryError.message(validation) }
    func limit(_ text: String, max: Int) throws -> StudioValue {
      let t = text.trimmingCharacters(in: .whitespaces)
      if t.isEmpty { return .null }
      guard let n = Int(t), n > 0, n <= max else {
        throw StudioLibraryError.message(
          "Enter a positive whole-number limit, or leave it empty for the default.")
      }
      return .number(Double(n))
    }
    return [
      "maxTokens": reply.value.map { .number(Double($0)) } ?? .null,
      "maxToolCalls": try limit(toolLimit, max: 10_000),
      "summarize": .bool(summarize),
      "mapTiles": .string(mapTiles),
    ]
  }
  func load(discard: Bool = false) async {
    guard !loading, !saving, discard || !dirty else { return }
    loading = true
    defer { loading = false }
    do {
      let result = try await command("preferencesGet", [:])
      try accept(result)
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  private func accept(_ reply: [String: StudioValue]) throws {
    guard let p = reply["preferences"]?.object,
      let summarize = p["summarize"]?.boolean, let tiles = p["mapTiles"]?.text,
      let layoutValue = p["layout"]
    else { throw StudioLibraryError.message("Studio preferences could not be read.") }
    let layout = try studioDecode(StudioSettingsLayout.self, layoutValue)
    guard layout.replyLimit.maximum == ReplyLimitDraft.maximum else {
      throw StudioLibraryError.message("Studio reply-limit settings could not be read.")
    }
    func limitText(_ key: String) throws -> String {
      if p[key] == .null { return "" }
      guard let value = p[key]?.number, let integer = Int(exactly: value), integer > 0 else {
        throw StudioLibraryError.message(
          "The saved \(key) preference is not a valid whole-number limit.")
      }
      return String(integer)
    }
    let reply = try limitText("maxTokens")
    let tools = try limitText("maxToolCalls")
    self.reply = ReplyLimitDraft(value: Int(reply))
    toolLimit = tools
    self.summarize = summarize
    self.layout = layout
    mapTiles = tiles
    mapHost = p["mapHost"]?.text ?? ""
    original = try values()
    loaded = true
  }
  func save() {
    guard loaded, !saving, !loading, dirty else { return }
    do {
      let next = try values()
      let changes = next.filter { original[$0.key] != $0.value }
      let expected = original.filter { changes[$0.key] != nil }
      saving = true
      error = nil
      notice = nil
      pending = Task {
        defer { self.saving = false }
        do {
          let result = try await self.command(
            "preferencesSave", ["changes": .object(changes), "expected": .object(expected)])
          try self.accept(result)
          self.notice = "Studio preferences saved."
        } catch { self.error = error.localizedDescription }
      }
    } catch { self.error = error.localizedDescription }
  }
  func settle() async { await pending?.value }
}
