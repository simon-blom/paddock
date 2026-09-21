import PaddockClient
import SwiftUI

/// Inline instance draft for an export selected in Catalog, not another model
/// browser or routed page. Nothing is saved or started until explicit submission.
struct StartModelView: View {
  let snapshot: ManagerSnapshot
  let purpose: ModelStartPurpose
  let onSubmit: (CreateEndpointRequest) async -> Bool
  let submissionError: () -> String?
  let onBrowse: (() -> Void)?
  let onDownload: ((String, String) -> Void)?
  let onClose: (() -> Void)?
  private let client: any ManagerLoading
  let initialModel: String
  let initialArtifact: String
  @Environment(\.dismiss) private var dismiss
  @State private var editor: EndpointEditor?
  @State private var tools: IntegrationsModel
  @State private var search: WebSearchEditor
  @State private var showsTools = false
  @State private var showsConnectors = false
  @State private var submitting = false
  @State private var loadingError: String?
  @State private var submitError: String?

  init(
    snapshot: ManagerSnapshot, model: String? = nil, artifact: String? = nil,
    purpose: ModelStartPurpose = .all, client: any ManagerLoading = NativeManager(),
    submissionError: @escaping () -> String? = { nil },
    onBrowse: (() -> Void)? = nil,
    onDownload: ((String, String) -> Void)? = nil,
    onClose: (() -> Void)? = nil,
    onSubmit: @escaping (CreateEndpointRequest) async -> Bool
  ) {
    self.snapshot = snapshot
    self.purpose = purpose
    self.client = client
    self.onSubmit = onSubmit
    self.submissionError = submissionError
    self.onBrowse = onBrowse
    self.onDownload = onDownload
    self.onClose = onClose
    let candidates = snapshot.catalog.models.filter {
      !purpose.weights($0, backend: snapshot.readiness.backend).isEmpty
    }
    // An explicit unavailable model/export never silently selects another.
    let selected: CatalogModel?
    if let model { selected = candidates.first { $0.id == model } } else { selected = nil }
    initialModel = selected?.id ?? ""
    initialArtifact =
      artifact ?? selected.flatMap {
        Self.preferredArtifact($0, snapshot: snapshot, purpose: purpose)?.id
      } ?? ""
    _tools = State(initialValue: IntegrationsModel(client: client))
    _search = State(
      initialValue: WebSearchEditor(
        client: client,
        settings: .init(port: 0, revision: "", provider: "", hasKey: false)))
  }

  static func artifacts(
    _ model: CatalogModel, snapshot: ManagerSnapshot, purpose: ModelStartPurpose = .all
  ) -> [CatalogArtifact] {
    purpose.weights(model, backend: snapshot.readiness.backend).filter(\.installed)
  }
  static func preferredArtifact(
    _ model: CatalogModel, snapshot: ManagerSnapshot, purpose: ModelStartPurpose = .all
  ) -> CatalogArtifact? {
    let installed = artifacts(model, snapshot: snapshot, purpose: purpose)
    return installed.first { $0.default == true } ?? installed.first
  }
  static func defaults(_ runtime: ArtifactRuntime?) -> (context: Int, batch: Int) {
    (min(runtime?.defaultMaxCtx ?? 32768, runtime?.memory?.maxCtx ?? 1_048_576), 1)
  }
  var speechUnavailable: Bool {
    purpose == .speech
      && !snapshot.catalog.models.contains {
        !purpose.weights($0, backend: snapshot.readiness.backend).isEmpty
      }
  }
  var validation: String? {
    guard let model = snapshot.catalog.models.first(where: { $0.id == initialModel }),
      Self.artifacts(model, snapshot: snapshot, purpose: purpose).contains(where: {
        $0.id == initialArtifact
      })
    else {
      return purpose == .speech
        ? "Select downloaded speech-to-text weights compatible with this Mac."
        : "Select a downloaded Metal weights artifact."
    }
    return nil
  }
  var request: CreateEndpointRequest? { editor?.creationRequest() }

  var body: some View {
    VStack(alignment: .leading, spacing: 22) {
      HStack {
        if showsTools {
          Button("Model settings", systemImage: "chevron.left") { showsTools = false }
            .buttonStyle(QuietButtonStyle())
        }
        Text(showsTools ? "System tools" : "New instance")
          .font(.system(size: 17, weight: .semibold))
        Spacer()
        Button("Cancel") { close() }.buttonStyle(QuietButtonStyle()).disabled(submitting)
          .accessibilityIdentifier("start-model-cancel")
      }
      VStack(alignment: .leading, spacing: 22) {
        if let validation {
          Text(validation).foregroundStyle(PaddockStyle.caution)
          Button("Choose in Catalog") { onBrowse?() }.buttonStyle(FlatButtonStyle())
            .disabled(onBrowse == nil)
        } else if let editor {
          if showsTools {
            EndpointFormCard("Web search") { WebSearchForm(editor: search) }
            EndpointMCPSection(model: tools, port: 0)
            if let error = tools.error {
              Text(error).foregroundStyle(PaddockStyle.caution)
              Button("Retry") { Task { await tools.refresh() } }.buttonStyle(QuietButtonStyle())
            }
            Button("Done") { showsTools = false }.buttonStyle(FlatButtonStyle(primary: true))
              .disabled(search.validation != nil)
          } else {
            EndpointSettingsView(
              editor: editor, canMutate: !submitting,
              onTools: { showsTools = true }, onCreate: start,
              onChangeModel: onBrowse,
              onDownload: { onDownload?(editor.modelID, $0) }
            )
            .task(id: "\(editor.modelID)/\(editor.artifactID)") {
              if editor.endpoint.model != editor.modelID
                || editor.endpoint.artifact != editor.artifactID
              {
                await editor.prepareCreationSelection()
              }
            }
            if let error = submitError {
              Text(error).foregroundStyle(PaddockStyle.caution).textSelection(.enabled)
            }
          }
        } else if let loadingError {
          Text(loadingError).foregroundStyle(PaddockStyle.caution)
          Button("Retry") { Task { await prepare() } }.buttonStyle(FlatButtonStyle())
        } else {
          ProgressView("Loading model settings…").controlSize(.small)
        }
      }.font(.system(size: 13))
        .disabled(submitting)
    }.frame(maxWidth: .infinity, alignment: .leading)
      .interactiveDismissDisabled(submitting)
      .task(id: "\(initialModel)/\(initialArtifact)/\(validation == nil)") { await prepare() }
      .onChange(
        of: snapshot.catalog.models.first { $0.id == (editor?.modelID ?? initialModel) }?
          .artifacts.map { "\($0.id)/\($0.installed)" }
      ) { _, _ in
        editor?.observeCatalog(snapshot.catalog.models)
      }
      .task {
        tools.onManage = { showsConnectors = true }
        await tools.refresh()
      }
      .sheet(
        isPresented: $showsConnectors, onDismiss: { Task { await tools.refresh() } },
        content: { StartConnectorLibrary(model: tools, endpoints: snapshot.servers ?? []) })
  }

  private func close() { if let onClose { onClose() } else { dismiss() } }

  private func prepare() async {
    guard validation == nil, editor == nil else { return }
    loadingError = nil
    let modelID = initialModel
    let artifactID = initialArtifact
    do {
      let value = try await client.prepareEndpoint(model: modelID, artifact: artifactID)
      guard !Task.isCancelled else { return }
      editor = EndpointEditor(
        client: client, endpoint: value, pid: nil,
        catalog: snapshot.catalog.models, isCreating: true, purpose: purpose)
    } catch { if !Task.isCancelled { loadingError = error.localizedDescription } }
  }

  private func start(networkConfirmed: Bool) {
    guard let editor, !submitting else { return }
    submitError = nil
    if let validation = search.validation, editor.canTools {
      submitError = validation
      showsTools = true
      return
    }
    let selected = tools.rows.filter { !$0.system && tools.endpointEnabled($0, port: 0) }
    let config =
      editor.canTools
      ? EndpointCreationTools(
        provider: search.provider, key: search.key,
        connectors: selected.map { .init(id: $0.id, revision: $0.revision) }) : nil
    guard let request = editor.creationRequest(networkConfirmed: networkConfirmed, tools: config)
    else { return }
    if let port = request.port,
      snapshot.runners.contains(where: { $0.port == port })
        || (snapshot.servers ?? []).contains(where: { $0.port == port })
    {
      submitError = "This port belongs to a saved model. Choose Automatic or another port."
      return
    }
    submitting = true
    Task {
      let accepted = await onSubmit(request)
      submitting = false
      if accepted {
        close()
      } else {
        submitError = submissionError() ?? "The model could not start. Your settings are kept."
      }
    }
  }
}

private struct StartConnectorLibrary: View {
  @Bindable var model: IntegrationsModel
  let endpoints: [ConfiguredEndpoint]
  @Environment(\.dismiss) private var dismiss
  var body: some View {
    VStack {
      HStack {
        Spacer()
        Button("Done") { dismiss() }.buttonStyle(QuietButtonStyle())
      }.padding(16)
      IntegrationsView(model: model, endpoints: endpoints)
    }.frame(width: 900, height: 620).background(PaddockStyle.canvas)
      .sheet(item: $model.editor) { editor in
        ConnectorReviewView(editor: editor, model: model, endpoints: endpoints) {
          Task { await model.cancelReview() }
        }
      }
  }
}
