import PaddockClient
import SwiftUI

/// Cloud and local catalogs share one edge-to-edge split. Service/account
/// controls belong to the list column; model/provider details own the right.
struct CloudModelsView: View {
  @Bindable var model: CloudBrowserModel
  var connections: ConnectionsModel? = nil
  @State var service: CloudService = .openrouter
  @State private var showsSaved = false

  static func canBrowse(_ service: CloudService, account: CloudConnection?) -> Bool {
    service == .openrouter
      || (account.map {
        ($0.hasKey || $0.allowUnauthenticated) && $0.credentialReady != false
      } ?? false)
  }

  var body: some View {
    let account = connections?.account(for: service)
    let available = Self.canBrowse(service, account: account)
    let browser =
      service != .openrouter && available
      ? account.flatMap { connections?.catalog(for: $0) } ?? model : model
    CloudCatalogView(
      model: browser, connections: connections, account: account, service: service,
      catalogAvailable: available,
      listHeader: AnyView(providerControls(account)),
      listFooter: AnyView(savedModels(account)),
      connectionPrompt: AnyView(connect(account))
    ).id("\(service.rawValue)/\(account?.id ?? "public")")
      .background(PaddockStyle.canvas).accessibilityIdentifier("cloud-browser")
      .task { await connections?.refresh() }
      .onChange(of: service) { _, _ in showsSaved = false }
  }

  private func providerControls(_ account: CloudConnection?) -> some View {
    VStack(alignment: .leading, spacing: 10) {
      Dropdown(title: "Provider", value: service.rawValue, fillsWidth: true, vendor: service.vendor)
      {
        ForEach(CloudService.allCases) { provider in
          Button {
            service = provider
          } label: {
            ModelProviderMenuLabel(title: provider.rawValue, vendor: provider.vendor)
          }
        }
      }.accessibilityIdentifier("cloud-provider-selector")
      if let connections {
        CloudAccountView(model: connections, service: service, account: account, compact: true)
      }
    }.padding(.horizontal, 16).padding(.top, 16)
      .accessibilityIdentifier("cloud-provider-section")
  }

  @ViewBuilder private func savedModels(_ account: CloudConnection?) -> some View {
    if let account, !account.models.isEmpty, let connections {
      VStack(spacing: 0) {
        WorkspaceRule()
        DisclosureGroup("Selected models (\(account.models.count))", isExpanded: $showsSaved) {
          CloudSavedPicksView(account: account, model: connections, compact: true)
            .padding(.top, 10)
        }.font(.system(size: 12, weight: .medium)).padding(16)
          .accessibilityIdentifier("cloud-selected-models")
      }
    }
  }

  private func connect(_ account: CloudConnection?) -> some View {
    let locked = account?.credentialReady == false
    return VStack(spacing: 14) {
      Image(systemName: locked ? "lock" : service == .custom ? "network" : "key")
        .font(.system(size: 26, weight: .light))
      Text(
        locked
          ? "Unlock \(service.rawValue)"
          : service == .custom
            ? "Connect your model server" : "Browse \(service.rawValue) models"
      )
      .font(.system(size: 19, weight: .medium))
      Button(locked ? "Unlock…" : service == .custom ? "Add endpoint…" : "Add API key…") {
        if locked, let account {
          Task { await connections?.unlock(account) }
        } else {
          connections?.review(connection: account, service: service)
        }
      }.buttonStyle(FlatButtonStyle(primary: true))
        .disabled(connections?.loaded != true || connections?.saving == true)
        .accessibilityIdentifier("cloud-connect")
    }.padding(32).frame(maxWidth: .infinity, maxHeight: .infinity)
  }
}
