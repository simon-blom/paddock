import PaddockClient
import SwiftUI

struct CloudAccountView: View {
  @Bindable var model: ConnectionsModel
  let service: CloudService
  let account: CloudConnection?
  var compact = false
  @State private var removing = false

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if compact, let account { accountPicker(account) }
      HStack(spacing: 10) {
        if let account {
          if !compact { accountPicker(account) }
          if model.checkingAccount == account.id {
            ProgressView().controlSize(.mini)
            Text("Testing connection…")
          } else if account.credentialReady == false {
            Label(compact ? "Locked" : "Cloud access locked", systemImage: "lock")
            Button("Unlock…") { Task { await model.unlock(account) } }
              .buttonStyle(FlatButtonStyle()).accessibilityIdentifier("cloud-unlock")
          } else if let check = model.checks[account.id] {
            Label(
              check.ok ? "Key works" : "Check failed",
              systemImage: check.ok ? "checkmark" : "exclamationmark.circle"
            )
            .help(check.message)
          } else {
            Text(
              account.allowUnauthenticated
                ? "No authentication · explicitly enabled"
                : account.hasKey ? "API key saved" : "No API key")
          }
          Spacer(minLength: 0)
          Menu {
            Button("Test connection") { Task { await model.test(account) } }
              .disabled(model.checkingAccount != nil)
            Button(
              service == .custom
                ? "Configure endpoint…" : account.hasKey ? "Replace key…" : "Add key…"
            ) {
              model.review(connection: account, service: service)
            }
            if service != .custom {
              Button("Connect another account…") { model.review(service: service) }
            }
            Divider()
            Button("Remove provider…") { removing = true }
          } label: {
            Image(systemName: "ellipsis").frame(width: 28, height: 28).contentShape(Rectangle())
          }.menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
            .accessibilityLabel("\(service.rawValue) settings").accessibilityIdentifier(
              "cloud-account-settings")
        } else {
          Text(service == .openrouter ? "Public catalog" : "No account connected")
          Spacer(minLength: 0)
          Button(service == .custom ? "Add endpoint…" : "Add key…") {
            model.review(service: service)
          }
          .buttonStyle(QuietButtonStyle()).disabled(!model.loaded)
        }
      }.font(.system(size: 11)).foregroundStyle(.secondary)
        .disabled(model.saving || model.hasDraft)
      if let error = model.error {
        HStack {
          Text(error).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
            .textSelection(.enabled)
          Spacer(minLength: 0)
          Button("Refresh accounts") { Task { await model.refresh() } }
            .buttonStyle(QuietButtonStyle()).font(.system(size: 11)).disabled(model.loading)
        }
      }
      if let account, let check = model.checks[account.id], !check.ok {
        Text(check.message).font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
      }
      if account?.credentialReady == false {
        Text("Unlock this account before chatting.")
          .font(.system(size: 11)).foregroundStyle(PaddockStyle.caution)
      }
      if let account, removing {
        VStack(alignment: .leading, spacing: 8) {
          Text("Remove \(account.name), its key and model picks? Chats will be kept.")
            .font(.system(size: 11))
          HStack {
            Button("Remove", role: .destructive) {
              Task {
                await model.remove(account)
                removing = false
              }
            }.buttonStyle(FlatButtonStyle()).disabled(model.saving)
            Button("Keep") { removing = false }.buttonStyle(FlatButtonStyle())
          }
        }
      }
    }.padding(.horizontal, compact ? 0 : 16).padding(.bottom, compact ? 0 : 8)
      .onChange(of: account?.id) { _, _ in removing = false }
  }

  @ViewBuilder private func accountPicker(_ account: CloudConnection) -> some View {
    let accounts = model.rows.filter { CloudService.service(for: $0) == service }
    if accounts.count > 1 || service == .custom {
      Dropdown(title: "Account", value: account.name, fillsWidth: compact) {
        ForEach(accounts) { row in
          Button(row.name) { model.selectedAccounts[service] = row.id }
        }
        Divider()
        Button(service == .custom ? "Add endpoint…" : "Connect another account…") {
          model.review(service: service)
        }
      }.accessibilityIdentifier("cloud-account").disabled(model.saving || model.hasDraft)
    }
  }
}

struct CloudSavedPicksView: View {
  let account: CloudConnection
  @Bindable var model: ConnectionsModel
  var compact = false
  var body: some View {
    VStack(alignment: .leading, spacing: 5) {
      if !compact {
        Text("In your pickers · \(account.models.count)")
          .font(.system(size: 12, weight: .semibold)).foregroundStyle(PaddockStyle.primary)
      }
      PaddockScrollView {
        LazyVStack(spacing: 0) {
          ForEach(account.models, id: \.pickKey) { pick in
            HStack(spacing: 10) {
              ModelAvatar(
                vendor: CloudCatalogPresentation.vendor(id: pick.id)
                  ?? CloudService.service(for: account).vendor, size: 18)
              VStack(alignment: .leading, spacing: 3) {
                Text(pick.display ?? pick.id).font(.system(size: 12)).lineLimit(1).help(pick.id)
                if compact, account.isOpenRouter {
                  Text(pick.provider ?? "Auto").font(.system(size: 10)).foregroundStyle(.secondary)
                    .lineLimit(1).help(pick.provider ?? "OpenRouter chooses the serving provider")
                }
              }
              Spacer(minLength: 0)
              if !compact, pick.vision == true { Image(systemName: "photo").help("Reads images") }
              if !compact, pick.reasoning == true { Image(systemName: "brain").help("Thinking") }
              if !compact, pick.asr == true { Image(systemName: "waveform").help("Speech to text") }
              if !compact, account.isOpenRouter {
                Text(pick.provider ?? "auto").font(.system(size: 10)).foregroundStyle(.secondary)
                  .lineLimit(1).help(
                    pick.provider.map { "Always routed to \($0); no fallback" }
                      ?? "OpenRouter chooses the provider per request")
              }
              if !compact, let ctx = pick.ctx {
                Text(CloudCatalogPresentation.tokens(ctx)).font(.system(size: 10)).monospacedDigit()
              }
              Button("Remove \(pick.display ?? pick.id)", systemImage: "xmark") {
                Task { await model.removeModel(pick, from: account) }
              }.labelStyle(.iconOnly).buttonStyle(QuietButtonStyle())
                .frame(width: 26, height: 28).disabled(model.saving || model.hasDraft)
                .accessibilityIdentifier("cloud-remove-pick-\(pick.pickKey)")
            }.font(.system(size: 10)).frame(height: compact ? 44 : 30)
          }
        }
      }.frame(height: min(compact ? 176 : 120, CGFloat(account.models.count) * (compact ? 44 : 30)))
        .scrollBounceBehavior(.basedOnSize)
    }.accessibilityIdentifier("cloud-saved-picks")
  }
}
