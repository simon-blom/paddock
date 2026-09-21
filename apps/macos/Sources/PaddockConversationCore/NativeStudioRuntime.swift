import Foundation

/// Single native owner of full conversation documents. The UI receives only a
/// projection; that projection is never used to overwrite the durable document.
/// No viewer, JavaScript engine, global cookie store or provider secret here.
public actor NativeStudioRuntime {
  public typealias V = ConversationValue
  public typealias O = [String: V]
  public typealias Presentation = @Sendable (O) async -> Void
  let transport: NativeConversationTransport
  let publish: Presentation
  let prepareGraph: @Sendable (O) async throws -> String
  let rasterPDF: @Sendable (Data, O) async throws -> [V]
  var graphGrounding = ""
  var document: ConversationDocument?
  var draft = true
  var models: [O] = []
  var caps: [String: O] = [:]
  var history: [O] = []
  var preferences: O = [:]
  var staged: [String: O] = [:]
  var artifacts: [V] = []
  var artifactRefreshEpoch = 0
  var artifactRefreshTask: Task<Void, Never>?
  var connectors: [O] = []
  var toolGroups: [O] = []
  var toolQuery = ""
  var documentPreview: O?
  var previewPart: O?
  var graphVisible = false
  var graphArtifact: O?
  var audio: O = [:]
  var search = "", sort = "newest", page = 1
  var draftText = "", error = ""
  var revision = 0
  var closed = false, mutating = false
  var tasks: [String: Task<Void, Never>] = [:]
  var stopRequested = false
  var reducers: [String: ResponseAccumulator] = [:]
  var starts: [String: ContinuousClock.Instant] = [:]
  var responseMetrics: [String: NativeResponseMetrics] = [:]
  var continuationPrefixes: [String: String] = [:]
  var completedTurn: V = .null
  var publication: Task<Void, Never>?
  var polling: Task<Void, Never>?
  var titleTask: Task<Void, Never>?
  var writeTail: Task<Void, Error>?
  var receipts: [String: O] = [:]
  var receiptOrder: [String] = []
  var lastCheckpoint = ContinuousClock.now

  public init(
    transport: NativeConversationTransport,
    prepareGraph: @escaping @Sendable (O) async throws -> String = { _ in
      throw ConversationFailure.invalid("A Traverse viewer is required for this graph")
    },
    rasterPDF: @escaping @Sendable (Data, O) async throws -> [V] = { _, _ in
      throw ConversationFailure.invalid("Native PDF rasterization is not available")
    },
    publish: @escaping Presentation
  ) {
    self.transport = transport
    self.publish = publish
    self.prepareGraph = prepareGraph
    self.rasterPDF = rasterPDF
  }
  public func start() async throws {
    let settings = try await transport.api("api/settings")
    preferences = settings["macos_studio_preferences"]?.object ?? [:]
    history = try await transport.listConversations()
    try await refreshModels()
    try newDocument()
    await emit()
    polling = Task { [weak self] in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(5))
        guard !Task.isCancelled, let self else { return }
        await self.poll()
      }
    }
  }
  func poll() async {
    guard !closed else { return }
    do {
      try await refreshModels()
      schedulePublish()
    } catch {
      // Keep the last known fleet; explicit Refresh reports failures.
    }
  }
  func newDocument() throws {
    let selected = document?.fields["model"]?.string
    let model =
      selected.flatMap { id in
        models.first { $0["id"]?.string == id && $0["status"]?.string == "ok" }
      }
      ?? models.first { $0["status"]?.string == "ok" && $0["kind"]?.string == "chat" }
      ?? models.first { $0["status"]?.string == "ok" }
    let now = Self.now
    document = try .init(fields: [
      "id": .string(UUID().uuidString), "title": .string("New conversation"),
      "model": model?["id"] ?? .string("default"), "messages": .array([]),
      "systemPrompt": .string(""),
      "params": .object(Self.defaultParams), "createdAt": now, "updatedAt": now,
      "kind": .string("chat"),
    ])
    draft = true
    staged = [:]
    draftText = ""
    documentPreview = nil
    previewPart = nil
    artifacts = []
    graphVisible = false
    graphArtifact = nil
    graphGrounding = ""
  }
  // The shared store indexes integer epoch milliseconds, as Date.now() does.
  static var now: V { .number(Decimal(Int64(Date().timeIntervalSince1970 * 1000))) }
  static let defaultParams: O = [
    "temperature": .null, "topP": .null, "topK": .null,
    "minP": .null, "presencePenalty": .null, "frequencyPenalty": .null, "repeatPenalty": .null,
    "seed": .null, "stop": .array([]), "thinking": .bool(true), "reasoningEffort": .string(""),
    "preserveThinking": .bool(false), "thinkingBudget": .null,
  ]
  var selected: [String] {
    let compare = document?.fields["compareModels"]?.array?.compactMap(\.string) ?? []
    if !compare.isEmpty { return compare }
    let id = document?.fields["model"]?.string ?? ""
    return id.isEmpty || id == "default" ? [] : [id]
  }
  var busy: Bool { mutating || !tasks.isEmpty }
  func idle() throws {
    guard !busy, (audio["phase"]?.string ?? "idle") == "idle" else {
      throw ConversationFailure.invalid(
        "Wait for the current operation, or stop the response first")
    }
  }
  func change(_ edit: (inout O) throws -> Void) throws {
    guard let doc = document else { throw ConversationFailure.invalid("Open a conversation first") }
    var fields = doc.fields
    try edit(&fields)
    document = try .init(fields: fields)
  }
  func updateMessage(_ id: String, _ edit: (inout O) -> Void) throws {
    try change { fields in
      var messages = fields["messages"]!.array!
      guard let index = messages.firstIndex(where: { $0["id"]?.string == id }),
        var m = messages[index].object
      else {
        throw ConversationFailure.stale
      }
      edit(&m)
      messages[index] = .object(m)
      fields["messages"] = .array(messages)
    }
  }
  /// Snapshot writes are queued in admission order, including Compare lanes.
  /// A failed terminal save leaves the full document in memory and a visible error.
  func save(_ doc: ConversationDocument) async throws {
    let previous = writeTail
    let transport = transport
    let task = Task {
      _ = try? await previous?.value
      try await transport.saveConversation(doc)
    }
    writeTail = task
    try await task.value
    history.removeAll { $0["id"]?.string == doc.id }
    history.append(doc.fields.filter { $0.key != "messages" })
  }
  func persist() async throws { if !draft, let document { try await save(document) } }
  func schedulePublish() {
    guard publication == nil, !closed else { return }
    publication = Task { [weak self] in
      try? await Task.sleep(for: .milliseconds(32))
      guard !Task.isCancelled, let self else { return }
      await self.emit()
    }
  }
  func emit() async {
    publication?.cancel()
    publication = nil
    guard !closed else { return }
    flushResponses()
    revision += 1
    await publish(presentation())
  }
  public func close() async {
    artifactRefreshTask?.cancel()
    artifactRefreshTask = nil
    closed = true
    polling?.cancel()
    publication?.cancel()
    titleTask?.cancel()
    let running = Array(tasks.values)
    for task in running { task.cancel() }
    for task in running { await task.value }
    _ = try? await writeTail?.value
    await transport.close()
  }
  public func setAudio(_ value: O) {
    audio = value
    schedulePublish()
  }
  public func currentFields() -> O? { document?.fields }
  public func viewerState() -> O {
    let graph = document?.activeMessages.flatMap { $0["content"]?.array ?? [] }.last {
      $0["type"]?.string == "graph"
    }
    return [
      "document": documentPreview.map(V.object) ?? .null,
      "graph": graphArtifact.map(V.object) ?? .null,
      "graphSource": graph ?? .null, "visibleGraph": .bool(graphVisible),
      "conversationId": document.map { .string($0.id) } ?? .null,
    ]
  }
}

extension ConversationValue {
  public var double: Double? {
    if case .number(let value) = self { return NSDecimalNumber(decimal: value).doubleValue }
    return nil
  }
}
