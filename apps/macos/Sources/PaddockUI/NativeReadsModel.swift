import CryptoKit
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
  struct Run: Identifiable, Codable {
    var id = UUID()
    var at = Date()
    let fingerprint: String
    let excerpt: String
    let characters: Int
    let questions: [ReadQuestion]
    let raw: ConversationValue
    let port: UInt16
    let elapsedMilliseconds: Double
    let response: ReadResponse
    enum CodingKeys: String, CodingKey {
      case id, at, fingerprint, excerpt, characters, questions, raw, port, elapsedMilliseconds
    }
    var retainedBytes: Int { (try? JSONEncoder().encode(self).count) ?? 0 }
    nonisolated static func fingerprint(_ value: ConversationValue) -> String {
      SHA256.hash(data: Data(((try? ReadDraft.json(value)) ?? "").utf8))
        .map { String(format: "%02x", $0) }.joined()
    }
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
  var stateError: String?
  var questionsError: String?
  var rowErrors: [String: String] = [:]
  var historyError: String?
  @ObservationIgnored var api: API
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored private var refreshError: String?
  @ObservationIgnored private var history: [String: [Run]] = [:]
  @ObservationIgnored private var loadedHistory: Set<String> = []
  @ObservationIgnored private var loadingHistory: Set<String> = []
  @ObservationIgnored private var historyEpoch: [String: Int] = [:]
  @ObservationIgnored private var latestRequest: ConversationValue?
  @ObservationIgnored private var latestRunID: UUID?
  @ObservationIgnored private var setsEpoch = 0
  @ObservationIgnored private var originalBody = ReadDraft().setBody
  @ObservationIgnored private var originalOrdering = ReadDraft().ordering
  @ObservationIgnored private var originalJSON = ""
  var hasUnappliedJSON: Bool { jsonText != originalJSON }
  var current: Reader? { readers.first { $0.port == port } }
  var result: Run? { runs.first { $0.id == selectedRun } ?? runs.first }
  var dirty: Bool {
    draft.setBody != originalBody || draft.ordering != originalOrdering
      || setName != (selectedSet?.name ?? "") || hasUnappliedJSON
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
    guard result.id == latestRunID else { return false }
    return result.port != port || latestRequest != draft.request(model: current?.model ?? "")
  }
  var previousRead: Bool { result != nil && result?.id != latestRunID }
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
      try Task.checkCancellation()
      readers = next
      if !readers.contains(where: { $0.port == port }) { port = readers.first?.port ?? 0 }
      let epoch = setsEpoch
      let saved = try await api("api/reads", "GET", nil, [:])
      if epoch == setsEpoch { sets = try decode([SavedSet].self, saved) }
      await restoreHistory(selectedSet?.id ?? "draft")
      if error == refreshError { error = nil }
      refreshError = nil
    } catch is CancellationError {} catch {
      refreshError = error.localizedDescription
      self.error = refreshError
    }
  }
  func add(_ kind: ReadQuestion.Kind) {
    guard draft.questions.count < (current?.maxQuestions ?? 64) else { return }
    var n = 1
    while draft.questions.contains(where: { $0.questionID == "q\(n)" }) { n += 1 }
    draft.questions.append(.init(questionID: "q\(n)", kind: kind))
  }
  func editID(_ id: UUID, text: String) {
    guard let index = draft.questions.firstIndex(where: { $0.id == id }) else { return }
    rowErrors[draft.questions[index].questionID] = nil
    draft.questions[index].idTouched = !text.isEmpty
    draft.questions[index].questionID =
      text.isEmpty
      ? ReadQuestion.derivedID(
        draft.questions[index].instructions,
        taken: draft.questions.filter { $0.id != id }.map(\.questionID))
      : ReadQuestion.cleanID(text)
  }
  func editInstructions(_ id: UUID, text: String) {
    guard let index = draft.questions.firstIndex(where: { $0.id == id }) else { return }
    rowErrors[draft.questions[index].questionID] = nil
    draft.questions[index].instructions = text
    if !draft.questions[index].idTouched {
      draft.questions[index].questionID = ReadQuestion.derivedID(
        text,
        taken: draft.questions.filter { $0.id != id }.map(\.questionID))
    }
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
    copy.idTouched = true
    let index = draft.questions.firstIndex { $0.id == id }!
    draft.questions.insert(copy, at: index + 1)
  }
  func move(_ id: UUID, by delta: Int) {
    guard let i = draft.questions.firstIndex(where: { $0.id == id }),
      draft.questions.indices.contains(i + delta)
    else { return }
    draft.questions.swapAt(i, i + delta)
  }
  @discardableResult func applyJSON(_ text: String) -> Bool {
    do {
      var parsed = try ReadDraft.parse(Data(text.utf8))
      // The questions tab and saved sets omit state; keep the loaded document.
      if try JSONDecoder().decode(ConversationValue.self, from: Data(text.utf8))["state"] == nil {
        parsed.state = draft.state
      }
      draft = parsed
      originalJSON = (try? draft.orderedJSON()) ?? ""
      jsonText = originalJSON
      error = nil
      questionsError = nil
      rowErrors = [:]
      return true
    } catch {
      questionsError = error.localizedDescription
      return false
    }
  }
  func beginJSON() {
    guard !hasUnappliedJSON else { return }
    originalJSON = (try? draft.orderedJSON()) ?? ""
    jsonText = originalJSON
  }
  func reset() {
    guard !busy && !saving && !importing else { return }
    draft = ReadDraft()
    jsonText = ""
    originalJSON = ""
    beginJSON()
    originalBody = draft.setBody
    originalOrdering = draft.ordering
    selectedSet = nil
    setName = ""
    fileName = ""
    runs = history["draft"] ?? []
    selectedRun = nil
    error = nil
    clearRunErrors()
    Task { await restoreHistory("draft") }
  }
  func open(_ set: SavedSet) {
    guard !busy && !saving && !importing else { return }
    do {
      var parsed = try ReadDraft.parse(Data(set.body.utf8))
      parsed.state = draft.state
      draft = parsed
      originalBody = parsed.setBody
      originalOrdering = parsed.ordering
      jsonText = ""
      originalJSON = ""
      beginJSON()
      selectedSet = set
      setName = set.name
      runs = history[set.id] ?? []
      selectedRun = nil
      error = nil
      clearRunErrors()
      Task { await restoreHistory(set.id) }
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
      let ordering = draft.ordering
      let serialized = try draft.orderedJSON()
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
      setsEpoch += 1
      sets = [saved] + sets.filter { $0.id != saved.id }
      selectedSet = saved
      originalBody = body
      originalOrdering = ordering
      if setName == submittedName { setName = saved.name }
      // Save As gives the current runs a new identity without losing them.
      if saved.id != previousID {
        history[saved.id] = history[previousID] ?? runs
        if previousID == "draft" { history[previousID] = nil }
        trimHistory()
        for run in history[saved.id] ?? [] { await persist(run, scope: saved.id) }
        loadedHistory.insert(saved.id)
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
      setsEpoch += 1
      historyEpoch[set.id, default: 0] += 1
      loadedHistory.insert(set.id)
      sets.removeAll { $0.id == set.id }
      history[set.id] = nil
      selectedSet = nil
      setName = ""
      runs = []
      selectedRun = nil
      error = nil
    } catch { self.error = error.localizedDescription }
  }
  func run(applyJSON: Bool = false) {
    if applyJSON, hasUnappliedJSON, !self.applyJSON(jsonText) { return }
    guard canRun, let reader = current else {
      if let validation { questionsError = validation }
      return
    }
    let request = draft.request(model: reader.model)
    guard draft.state.utf8.count <= 4 * 1024 * 1024 else {
      stateError = "The text exceeds the 4 MiB reading limit. Use a smaller section."
      return
    }
    let questions = draft.questions
    let setID = selectedSet?.id ?? "draft"
    busy = true
    error = nil
    clearRunErrors()
    task = Task {
      defer { busy = false }
      do {
        let started = ContinuousClock.now
        let raw = try await api("api/runners/\(reader.port)/v1/systemone", "POST", request, [:])
        try Task.checkCancellation()
        let response = try decode(ReadResponse.self, raw)
        try response.validate(for: questions)
        let elapsed = started.duration(to: .now)
        let milliseconds =
          Double(elapsed.components.seconds) * 1000
          + Double(elapsed.components.attoseconds) / 1e15
        let fingerprint = await Task.detached(priority: .utility) { Run.fingerprint(request) }.value
        try Task.checkCancellation()
        let result = Run(
          fingerprint: fingerprint,
          excerpt: String(
            (request["state"]?.string ?? "").split(whereSeparator: \.isWhitespace)
              .joined(separator: " ").prefix(120)),
          characters: request["state"]?.string?.count ?? 0,
          questions: questions, raw: raw, port: reader.port, elapsedMilliseconds: milliseconds,
          response: response)
        history[setID] = Array(([result] + (history[setID] ?? [])).prefix(10))
        latestRequest = request
        latestRunID = result.id
        trimHistory()
        runs = history[setID] ?? []
        selectedRun = result.id
        await persist(result, scope: setID)
      } catch is CancellationError {} catch { routeError(error.localizedDescription) }
    }
  }
  func cancel() { task?.cancel() }
  func runExample() {
    guard !busy, !saving, !importing else { return }
    reset()
    draft = .example
    beginJSON()
    run()
  }
  // Bounded results plus one volatile request for edit detection. Durable
  // history contains only a fingerprint and excerpt, never the full state.
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
    if let latestRunID, !keep.contains(latestRunID) {
      latestRequest = nil
      self.latestRunID = nil
    }
  }
  func settle() async { await task?.value }
  func clearRunErrors() {
    stateError = nil
    questionsError = nil
    rowErrors = [:]
  }
  func routeError(_ message: String) {
    if message.hasPrefix("question \""),
      let expression = try? NSRegularExpression(pattern: #"^question \"((?:[^\"\\]|\\.)+)\":"#),
      let match = expression.firstMatch(
        in: message, range: NSRange(message.startIndex..., in: message)),
      let range = Range(match.range(at: 1), in: message)
    {
      let raw = String(message[range])
      let id = (try? JSONDecoder().decode(String.self, from: Data(("\"" + raw + "\"").utf8))) ?? raw
      if draft.questions.contains(where: { $0.questionID == id }) {
        rowErrors[id] = message
      } else {
        error = message
      }
    } else if message.hasPrefix("state:") || message.contains("the window is") {
      stateError = message
    } else if message.hasPrefix("questions:") || message.hasPrefix("samples:")
      || message.contains("the answer template needs")
    {
      questionsError = message
    } else {
      error = message
    }
  }
  private func persist(_ run: Run, scope: String) async {
    do {
      let value = try JSONDecoder().decode(ConversationValue.self, from: JSONEncoder().encode(run))
      _ = try await api("api/read-runs/\(scope)", "POST", value, [:])
      historyError = nil
    } catch {
      historyError =
        "The result is available, but history could not be saved: \(error.localizedDescription)"
    }
  }
  func restoreHistory(_ scope: String) async {
    guard !loadedHistory.contains(scope), !loadingHistory.contains(scope) else { return }
    loadingHistory.insert(scope)
    let epoch = historyEpoch[scope, default: 0]
    defer { loadingHistory.remove(scope) }
    do {
      let value = try await api("api/read-runs/\(scope)", "GET", nil, [:])
      let saved = try decode([Run].self, value)
      guard epoch == historyEpoch[scope, default: 0] else { return }
      for run in saved {
        try run.response.validate(for: run.questions)
      }
      let local = history[scope] ?? []
      let ids = Set(local.map(\.id))
      history[scope] = Array(
        (local + saved.filter { !ids.contains($0.id) })
          .sorted { $0.at > $1.at }.prefix(10))
      loadedHistory.insert(scope)
      trimHistory()
      if (selectedSet?.id ?? "draft") == scope { runs = history[scope] ?? [] }
      historyError = nil
    } catch is CancellationError {} catch {
      historyError = "Read history could not be loaded: \(error.localizedDescription)"
    }
  }
  func clearHistory() async {
    guard !busy, !saving else { return }
    let scope = selectedSet?.id ?? "draft"
    do {
      _ = try await api("api/read-runs/\(scope)", "DELETE", nil, [:])
      historyEpoch[scope, default: 0] += 1
      loadedHistory.insert(scope)
      history[scope] = nil
      runs = []
      selectedRun = nil
      latestRequest = nil
      latestRunID = nil
      historyError = nil
    } catch { historyError = error.localizedDescription }
  }
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

extension NativeReadsModel.Run {
  init(from decoder: any Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(UUID.self, forKey: .id)
    at = try c.decode(Date.self, forKey: .at)
    fingerprint = try c.decode(String.self, forKey: .fingerprint)
    excerpt = try c.decode(String.self, forKey: .excerpt)
    characters = try c.decode(Int.self, forKey: .characters)
    questions = try c.decode([ReadQuestion].self, forKey: .questions)
    raw = try c.decode(ConversationValue.self, forKey: .raw)
    port = try c.decode(UInt16.self, forKey: .port)
    elapsedMilliseconds = try c.decode(Double.self, forKey: .elapsedMilliseconds)
    response = try JSONDecoder().decode(ReadResponse.self, from: JSONEncoder().encode(raw))
  }
}
