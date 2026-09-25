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
    if path.hasPrefix("api/read-history/") && method == "GET" {
      // A 16 MiB document expands when carried as a JSON string envelope.
      let data = try await transport!.bytes(
        path, method: method, query: query, maximum: 40 * 1024 * 1024)
      return try JSONDecoder().decode(ConversationValue.self, from: data)
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
    var state: String?
    var fileName = ""
    var samples = 0
    enum CodingKeys: String, CodingKey {
      case id, at, fingerprint, excerpt, characters, questions, raw, port, elapsedMilliseconds,
        state, fileName, samples
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
  struct Session: Decodable, Identifiable {
    let id: String
    let title: String
    let model: String
    let runs: Int
    let updatedAt: Double
  }
  private(set) var sessions: [Session] = []
  private(set) var activeSession: ReadHistoryDocument?
  private var sessionRevision = ""
  private var sessionEpoch = 0
  private(set) var openingSession = false
  private(set) var historyUnsaved = false
  @ObservationIgnored var api: API
  @ObservationIgnored private var task: Task<Void, Never>?
  @ObservationIgnored private var refreshError: String?
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
  var hasWork: Bool {
    dirty || busy || saving || importing || historyUnsaved || !draft.state.isEmpty
  }
  var validation: String? {
    draft.validation(
      maxQuestions: current?.maxQuestions ?? 64, maxSamples: current?.maxSamples ?? 32)
      ?? (draft.questions.contains {
        !(current?.types ?? ["noul", "choice", "score"]).contains($0.kind.rawValue)
      }
        ? "This runner does not support one of these question types." : nil)
  }
  var canRun: Bool {
    current != nil && !busy && !saving && !importing && !openingSession && validation == nil
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
      await refreshHistory()
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
    sessionEpoch += 1
    activeSession = nil
    sessionRevision = ""
    openingSession = false
    historyUnsaved = false
    runs = []
    latestRequest = nil
    latestRunID = nil
    selectedRun = nil
    error = nil
    clearRunErrors()
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
      error = nil
      clearRunErrors()
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
      sets.removeAll { $0.id == set.id }
      selectedSet = nil
      setName = ""
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
    let submittedFileName = fileName
    let submittedSamples = draft.samples
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
          response: response, state: request["state"]?.string,
          fileName: submittedFileName, samples: submittedSamples)
        latestRequest = request
        latestRunID = result.id
        runs = Array(([result] + runs).prefix(20))
        selectedRun = result.id
        await keepRun(result)
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
  // Bound in-memory results. Durable history is the shared SQLite document.
  func trimHistory(maxBytes: Int = 16 * 1024 * 1024) {
    var used = 0
    let keep = Set(
      runs.sorted { $0.at > $1.at }.prefix { run in
        used += run.retainedBytes
        return used <= maxBytes
      }.map(\.id))
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
  func refreshHistory() async {
    let epoch = sessionEpoch
    do {
      let rows = try await api("api/read-history", "GET", nil, [:])
      if epoch == sessionEpoch { sessions = try decode([Session].self, rows) }
    } catch is CancellationError {} catch {
      historyError = "Read history could not be loaded: \(error.localizedDescription)"
    }
  }

  private func keepRun(_ run: Run) async {
    let at = (run.at.timeIntervalSince1970 * 1000).rounded()
    let value: ConversationValue = .object([
      "id": .string(run.id.uuidString), "at": .number(Decimal(at)),
      "model": .string(run.response.model),
      "port": .number(Decimal(run.port)), "excerpt": .string(run.excerpt),
      "chars": .number(Decimal(run.characters)), "state": .string(run.state ?? ""),
      "fileName": .string(run.fileName),
      "questions": .object(
        Dictionary(
          run.questions.map { ($0.questionID, $0.wire) }, uniquingKeysWith: { _, b in b })),
      "questionOrder": .array(
        run.questions.map { q in
          .array(
            ([q.questionID]
              + (q.kind == .choice
                ? q.options.map { $0.name.trimmingCharacters(in: .whitespacesAndNewlines) } : []))
              .map(ConversationValue.string))
        }),
      "samples": run.samples == 0 ? .string("auto") : .number(Decimal(run.samples)),
      "response": run.raw, "ms": .number(Decimal(run.elapsedMilliseconds)),
    ])
    let title =
      run.fileName.isEmpty
      ? String(
        (run.state ?? "").split(separator: "\n").first.map(String.init)?.prefix(60)
          ?? "Untitled read")
      : run.fileName
    var doc =
      activeSession?.value.object ?? [
        "id": .string(UUID().uuidString), "title": .string(title),
        "createdAt": .number(Decimal(at)),
      ]
    doc["updatedAt"] = .number(Decimal(at))
    doc["model"] = value["model"]
    doc["runs"] = .array(Array(((doc["runs"]?.array ?? []) + [value]).suffix(20)))
    activeSession = ReadHistoryDocument(value: .object(doc))
    historyUnsaved = true
    await saveHistory()
  }

  func saveHistory() async {
    guard !saving, let doc = activeSession else { return }
    saving = true
    defer { saving = false }
    let epoch = sessionEpoch
    do {
      let json = try await Task.detached(priority: .utility) { try doc.json }.value
      guard json.utf8.count <= 16 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
      let reply = try await api(
        "api/read-history/\(doc.id)", "PUT",
        .object(["doc": .string(json)]), ["envelope": "true", "revision": sessionRevision])
      guard epoch == sessionEpoch else { return }
      guard let revision = reply["read"]?["revision"]?.string else {
        throw ConversationFailure.invalid("Read history was not acknowledged.")
      }
      sessionRevision = revision
      historyUnsaved = false
      historyError = nil
      await refreshHistory()
    } catch is CancellationError {} catch {
      historyError =
        "The result is available, but history could not be saved: \(error.localizedDescription)"
    }
  }

  func openSession(_ id: String) async {
    guard !busy, !saving, !importing, ConversationDocument.validID(id) else { return }
    sessionEpoch += 1
    let epoch = sessionEpoch
    let before = draft
    openingSession = true
    defer { if epoch == sessionEpoch { openingSession = false } }
    do {
      let snapshot = try await api("api/read-history/\(id)", "GET", nil, ["envelope": "true"])
      guard let text = snapshot["doc"]?.string, let revision = snapshot["revision"]?.string else {
        throw ConversationFailure.invalid("Invalid read history response.")
      }
      let doc = try await Task.detached(priority: .userInitiated) {
        try ReadHistoryDocument(json: text)
      }.value
      guard doc.id == id else {
        throw ConversationFailure.invalid("The read identity does not match its address.")
      }
      guard epoch == sessionEpoch, draft == before else { return }
      var restored: [Run] = []
      for (index, value) in doc.runs.enumerated() {
        let input = try ReadHistoryDocument.draft(value)
        let raw = value["response"] ?? .null
        let response = try decode(ReadResponse.self, raw)
        try response.validate(for: input.questions)
        guard let p = value["port"]?.integer, let runPort = UInt16(exactly: p), runPort > 0 else {
          throw ConversationFailure.invalid("Invalid saved reader port.")
        }
        var run = Run(
          fingerprint: "", excerpt: value["excerpt"]?.string ?? "",
          characters: value["chars"]?.integer ?? 0, questions: input.questions, raw: raw,
          port: runPort, elapsedMilliseconds: try decode(Double.self, value["ms"] ?? .number(0)),
          response: response,
          state: value["stateMissing"] == .bool(true) ? nil : input.state,
          fileName: value["fileName"]?.string ?? "", samples: input.samples)
        // Legacy web runs have no UUID. Stable within this loaded document.
        run.id = value["id"]?.string.flatMap(UUID.init(uuidString:)) ?? UUID()
        run.at = Date(
          timeIntervalSince1970: try decode(Double.self, value["at"] ?? .number(0)) / 1000)
        restored.append(run)
        if index == doc.runs.count - 1 {
          draft = input
          fileName = run.fileName
          if readers.contains(where: { $0.port == runPort }) { port = runPort }
        }
      }
      activeSession = doc
      sessionRevision = revision
      runs = restored.reversed()
      selectedRun = runs.first?.id
      latestRunID = runs.first?.id
      latestRequest = draft.request(model: runs.first?.response.model ?? "")
      selectedSet = nil
      setName = ""
      jsonText = ""
      originalJSON = ""
      beginJSON()
      originalBody = draft.setBody
      originalOrdering = draft.ordering
      historyUnsaved = false
      historyError = nil
      clearRunErrors()
    } catch is CancellationError {} catch { historyError = error.localizedDescription }
  }

  func clearHistory() async {
    guard !busy, !saving, let doc = activeSession else { return }
    saving = true
    defer { saving = false }
    do {
      _ = try await api("api/read-history/\(doc.id)", "DELETE", nil, ["revision": sessionRevision])
      sessionEpoch += 1
      sessions.removeAll { $0.id == doc.id }
      activeSession = nil
      sessionRevision = ""
      openingSession = false
      historyUnsaved = false
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
    state = try c.decodeIfPresent(String.self, forKey: .state)
    fileName = try c.decodeIfPresent(String.self, forKey: .fileName) ?? ""
    samples = try c.decodeIfPresent(Int.self, forKey: .samples) ?? 0
    response = try JSONDecoder().decode(ReadResponse.self, from: JSONEncoder().encode(raw))
  }
}
