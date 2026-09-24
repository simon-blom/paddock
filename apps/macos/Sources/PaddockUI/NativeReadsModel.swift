import Foundation
import Observation
import PaddockClient
import PaddockConversationCore
import UniformTypeIdentifiers

private actor ReadsConnection {
  let client: any ManagerLoading
  var transport: NativeConversationTransport?
  init(client: any ManagerLoading) { self.client = client }
  func call(_ path: String, _ method: String, _ body: ConversationValue?, _ query: [String: String])
    async throws -> ConversationValue
  {
    if transport == nil {
      let host = try await client.nativeConversationHost()
      transport = try NativeConversationTransport(host: host)
    }
    return try await transport!.api(path, method: method, body: body, query: query)
  }
}

@MainActor @Observable final class NativeReadsModel {
  typealias API =
    @MainActor (String, String, ConversationValue?, [String: String]) async throws ->
    ConversationValue
  struct Reader: Identifiable, Equatable {
    let port: UInt16
    let model: String
    let title: String
    let vendor: String?
    let maxQuestions: Int
    let maxSamples: Int
    let types: [String]
    var id: UInt16 { port }
  }
  struct SavedSet: Decodable, Identifiable, Equatable {
    let id: String
    let name: String
    let body: String
    let revision: String
  }
  struct Run: Identifiable {
    let id = UUID()
    let at = Date()
    let request: ConversationValue
    let questions: [ReadQuestion]
    let response: ReadResponse
    let raw: ConversationValue
    let port: UInt16
    let retainedBytes: Int
  }
  var draft = ReadDraft()
  var port: UInt16 = 0
  var setName = ""
  var jsonText = ""
  var fileName = ""
  private(set) var readers: [Reader] = []
  private(set) var sets: [SavedSet] = []
  private(set) var selectedSet: SavedSet?
  private(set) var runs: [Run] = []
  var selectedRun: UUID?
  private(set) var loading = false
  private(set) var busy = false
  private(set) var saving = false
  private(set) var importing = false
  var error: String?
  @ObservationIgnored var api: API
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored private var history: [String: [Run]] = [:]
  @ObservationIgnored private var originalBody = ReadDraft().setBody
  @ObservationIgnored private var originalJSON = ""
  var hasUnappliedJSON: Bool { jsonText != originalJSON }
  var current: Reader? { readers.first { $0.port == port } }
  var result: Run? { runs.first { $0.id == selectedRun } ?? runs.first }
  var dirty: Bool {
    draft.setBody != originalBody || setName != (selectedSet?.name ?? "") || hasUnappliedJSON
  }
  var hasWork: Bool { dirty || busy || saving || importing || !draft.state.isEmpty }
  var validation: String? {
    draft.validation(
      maxQuestions: current?.maxQuestions ?? 64, maxSamples: current?.maxSamples ?? 32)
      ?? (draft.questions.contains {
        !(current?.types ?? ["noul", "choice", "score"]).contains($0.kind.rawValue)
      }
        ? "This runner does not support one of these question types." : nil)
  }
  var canRun: Bool {
    current != nil && !busy && !importing && validation == nil
      && !draft.state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
  }
  var stale: Bool {
    guard let result else { return false }
    return result.port != port || result.request != draft.request(model: current?.model ?? "")
  }
  init(client: any ManagerLoading) {
    let connection = ReadsConnection(client: client)
    api = { path, method, body, query in try await connection.call(path, method, body, query) }
  }
  private func decode<T: Decodable>(_ type: T.Type, _ v: ConversationValue) throws -> T {
    try JSONDecoder().decode(type, from: JSONEncoder().encode(v))
  }
  func refresh() async {
    guard !loading else { return }
    loading = true
    defer { loading = false }
    do {
      let fleet = try await api("api/runners", "GET", nil, [:]).array ?? []
      var next: [Reader] = []
      for runner in fleet {
        try Task.checkCancellation()
        guard let number = runner["port"]?.integer, let port = UInt16(exactly: number), port > 0,
          let id = runner["model"]?.string
        else { continue }
        do {
          let info = try await api("api/runners/\(port)/server", "GET", nil, [:])
          guard let caps = info["structured_read"], (caps["canvas_width"]?.integer ?? 0) > 0 else {
            continue
          }
          next.append(
            Reader(
              port: port, model: id, title: runner["display"]?.string ?? id,
              vendor: runner["vendor"]?.string,
              maxQuestions: min(64, max(1, caps["max_questions"]?.integer ?? 64)),
              maxSamples: min(32, max(1, caps["max_samples"]?.integer ?? 32)),
              types: caps["types"]?.array?.compactMap(\.string) ?? ["noul", "choice", "score"]))
        } catch is CancellationError { throw CancellationError() } catch {
          if let previous = readers.first(where: { $0.port == port && $0.model == id }) {
            next.append(previous)
          }
        }
      }
      let saved = try await api("api/reads", "GET", nil, [:])
      let records = try decode([SavedSet].self, saved)
      try Task.checkCancellation()
      readers = next
      sets = records
      if !readers.contains(where: { $0.port == port }) { port = readers.first?.port ?? 0 }
    } catch is CancellationError {} catch { self.error = error.localizedDescription }
  }
  func add(_ kind: ReadQuestion.Kind) {
    guard draft.questions.count < (current?.maxQuestions ?? 64) else { return }
    var n = 1
    while draft.questions.contains(where: { $0.questionID == "q\(n)" }) { n += 1 }
    draft.questions.append(.init(questionID: "q\(n)", kind: kind))
  }
  func duplicate(_ id: UUID) {
    guard let row = draft.questions.first(where: { $0.id == id }),
      draft.questions.count < (current?.maxQuestions ?? 64)
    else { return }
    var copy = row
    copy.id = UUID()
    var n = 2
    while draft.questions.contains(where: { $0.questionID == "\(row.questionID)_\(n)" }) { n += 1 }
    copy.questionID = "\(row.questionID)_\(n)"
    draft.questions.append(copy)
  }
  func move(_ id: UUID, by delta: Int) {
    guard let i = draft.questions.firstIndex(where: { $0.id == id }),
      draft.questions.indices.contains(i + delta)
    else { return }
    draft.questions.swapAt(i, i + delta)
  }
  func applyJSON(_ text: String) {
    do {
      var parsed = try ReadDraft.parse(Data(text.utf8))
      // The questions tab and saved sets omit state; keep the loaded document.
      if try JSONDecoder().decode(ConversationValue.self, from: Data(text.utf8))["state"] == nil {
        parsed.state = draft.state
      }
      draft = parsed
      originalJSON = (try? ReadDraft.json(draft.setBody)) ?? ""
      jsonText = originalJSON
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func beginJSON() {
    guard !hasUnappliedJSON else { return }
    originalJSON = (try? ReadDraft.json(draft.setBody)) ?? ""
    jsonText = originalJSON
  }
  func reset() {
    guard !busy && !saving && !importing else { return }
    draft = ReadDraft()
    jsonText = ""
    originalJSON = ""
    beginJSON()
    originalBody = draft.setBody
    selectedSet = nil
    setName = ""
    fileName = ""
    runs = history["draft"] ?? []
    selectedRun = nil
    error = nil
  }
  func open(_ set: SavedSet) {
    guard !busy && !saving && !importing else { return }
    do {
      var parsed = try ReadDraft.parse(Data(set.body.utf8))
      parsed.state = draft.state
      draft = parsed
      originalBody = parsed.setBody
      jsonText = ""
      originalJSON = ""
      beginJSON()
      selectedSet = set
      setName = set.name
      runs = history[set.id] ?? []
      selectedRun = nil
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func save(asNew: Bool = false) async {
    guard !saving, !busy, validation == nil else { return }
    guard !hasUnappliedJSON else {
      error = "Apply the JSON questions before saving."
      return
    }
    let name = setName.trimmingCharacters(in: .whitespacesAndNewlines)
    let submittedName = setName
    let previousID = selectedSet?.id ?? "draft"
    guard !name.isEmpty && name.utf8.count <= 512 else {
      error = "Enter a set name up to 512 bytes."
      return
    }
    saving = true
    defer { saving = false }
    do {
      let body = draft.setBody
      let serialized = try ReadDraft.json(body)
      guard serialized.utf8.count <= 512 * 1024 else { throw ConversationFailure.tooLarge }
      let record: ConversationValue = .object([
        "id": .string(asNew ? UUID().uuidString : selectedSet?.id ?? UUID().uuidString),
        "name": .string(name), "body": .string(serialized),
        "revision": .string(asNew ? "" : selectedSet?.revision ?? ""),
      ])
      let reply = try await api("api/reads", "POST", record, [:])
      guard let value = reply["set"] else {
        throw ConversationFailure.invalid("The saved set was not acknowledged.")
      }
      let saved = try decode(SavedSet.self, value)
      sets = [saved] + sets.filter { $0.id != saved.id }
      selectedSet = saved
      originalBody = body
      if setName == submittedName { setName = saved.name }
      // Save As gives the current runs a new identity without losing them.
      if saved.id != previousID {
        history[saved.id] = history[previousID] ?? runs
        if previousID == "draft" { history[previousID] = nil }
        trimHistory()
      }
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func remove() async {
    guard !saving, !busy, let set = selectedSet else { return }
    saving = true
    defer { saving = false }
    do {
      guard ConversationDocument.validID(set.id) else {
        throw ConversationFailure.invalid("Invalid saved set identity.")
      }
      _ = try await api("api/reads/\(set.id)", "DELETE", nil, ["revision": set.revision])
      sets.removeAll { $0.id == set.id }
      history[set.id] = nil
      selectedSet = nil
      setName = ""
      runs = []
      selectedRun = nil
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func run() {
    guard canRun, let reader = current else { return }
    let request = draft.request(model: reader.model)
    let questions = draft.questions
    let setID = selectedSet?.id ?? "draft"
    busy = true
    error = nil
    task = Task {
      defer { busy = false }
      do {
        let raw = try await api("api/runners/\(reader.port)/v1/systemone", "POST", request, [:])
        try Task.checkCancellation()
        let response = try decode(ReadResponse.self, raw)
        guard Set(response.answers.keys) == Set(questions.map(\.questionID)) else {
          throw ConversationFailure.invalid("The runner returned an incomplete set of answers.")
        }
        let bytes = try JSONEncoder().encode(request).count + JSONEncoder().encode(raw).count
        let result = Run(
          request: request, questions: questions, response: response, raw: raw,
          port: reader.port, retainedBytes: bytes)
        history[setID] = Array(([result] + (history[setID] ?? [])).prefix(10))
        trimHistory()
        runs = history[setID] ?? []
        selectedRun = result.id
      } catch is CancellationError {} catch { self.error = error.localizedDescription }
    }
  }
  func cancel() { task?.cancel() }
  // Full document snapshots are useful, but ten runs for each of many sets
  // must not retain hundreds of MiB. Global LRU bound, not a per-set reset.
  func trimHistory(maxBytes: Int = 16 * 1024 * 1024) {
    var used = 0
    let keep = Set(
      history.values.flatMap { $0 }.sorted { $0.at > $1.at }.prefix { run in
        used += run.retainedBytes
        return used <= maxBytes
      }.map(\.id))
    for key in Array(history.keys) {
      let retained = history[key, default: []].filter { keep.contains($0.id) }
      history[key] = retained.isEmpty ? nil : retained
    }
    runs = runs.filter { keep.contains($0.id) }
  }
  func settle() async { await task?.value }
  func loadFile(_ url: URL, asJSON: Bool = false) async {
    guard !importing && !busy else { return }
    importing = true
    defer { importing = false }
    let old = draft.state
    do {
      let bytes = try await Task.detached(priority: .userInitiated) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let cap = asJSON ? 512 * 1024 : 32 * 1024 * 1024
        let data = try file.read(upToCount: cap + 1) ?? Data()
        guard data.count <= cap else { throw ConversationFailure.tooLarge }
        return data
      }.value
      try Task.checkCancellation()
      if asJSON {
        applyJSON(String(decoding: bytes, as: UTF8.self))
        return
      }
      let ext = url.pathExtension.lowercased()
      let plain = [
        "txt", "md", "markdown", "csv", "tsv", "json", "log", "xml", "html", "htm", "yaml", "yml",
        "toml", "eml",
      ]
      let text: String
      let decoded: String?
      if plain.contains(ext) {
        decoded = await Task.detached(priority: .userInitiated) {
          String(data: bytes, encoding: .utf8)
        }.value
      } else {
        decoded = nil
      }
      if let utf8 = decoded {
        text = utf8
      } else {
        guard let reader = current else {
          throw ConversationFailure.invalid("Start a reading model before extracting this file.")
        }
        let mime = UTType(filenameExtension: ext)?.preferredMIMEType ?? "application/octet-stream"
        let encoded = await Task.detached(priority: .userInitiated) {
          "data:\(mime);base64,\(bytes.base64EncodedString())"
        }.value
        try Task.checkCancellation()
        let reply = try await api(
          "api/runners/\(reader.port)/extract", "POST",
          .object([
            "filename": .string(url.lastPathComponent),
            "data": .string(encoded),
            "file_metadata": .string("off"),
          ]), [:])
        guard let value = reply["text"]?.string else {
          throw ConversationFailure.invalid("The file did not produce readable text.")
        }
        text = value
      }
      guard draft.state == old else {
        throw ConversationFailure.invalid(
          "The text changed while the file was loading. Your edits were kept.")
      }
      guard text.utf8.count <= 4 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
      draft.state = text
      fileName = url.lastPathComponent
      error = nil
    } catch is CancellationError {} catch { self.error = error.localizedDescription }
  }
}
