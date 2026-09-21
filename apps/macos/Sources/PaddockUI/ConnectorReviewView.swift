import PaddockClient
import SwiftUI

struct ConnectorReviewView: View {
  @Bindable var editor: ConnectorEditor
  var model: IntegrationsModel? = nil
  var endpoints: [ConfiguredEndpoint] = []
  let onCancel: () -> Void
  @State private var toolQuery = ""
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      Text(editor.original == nil ? "Add connector" : "Edit connector").font(
        .system(size: 20, weight: .semibold)
      ).padding(22)
      WorkspaceRule()
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 18) {
          if editor.authHint {
            Text(
              "This server needs credentials. Add a header below or save it and connect your account."
            )
            .font(.system(size: 12)).foregroundStyle(.secondary)
          }
          IntegrationField("Name") { TextField("e.g. research", text: $editor.draft.label) }
          IntegrationField("MCP server URL") { TextField("https://…/mcp", text: $editor.draft.url) }
          Dropdown(
            title: "Credentials",
            value: [
              "keep": "Keep saved headers", "none": "No credential headers",
              "header": "Replace credential headers",
            ][editor.credentialMode] ?? "Credentials", fillsWidth: true
          ) {
            if editor.original != nil {
              Button("Keep saved headers") { editor.credentialMode = "keep" }
            }
            Button("No credential headers") { editor.credentialMode = "none" }
            Button("Replace credential headers") { editor.credentialMode = "header" }
          }
          if editor.credentialMode == "header" {
            ForEach($editor.headerFields) { $header in
              HStack(alignment: .bottom, spacing: 10) {
                VStack(alignment: .leading, spacing: 10) {
                  IntegrationField("Header name") { TextField("Authorization", text: $header.name) }
                  IntegrationField("Header value") {
                    SecureField("Bearer … or the provider's API key", text: $header.value)
                  }
                }
                Button("Remove header", systemImage: "minus.circle") {
                  editor.headerFields.removeAll { $0.id == header.id }
                }.labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).disabled(
                  editor.headerFields.count == 1)
              }
            }
            Button("Add header", systemImage: "plus") {
              editor.headerFields.append(.init(name: "", value: ""))
            }.buttonStyle(QuietButtonStyle()).disabled(editor.headerFields.count >= 16)
            Text(
              "Replaces all saved headers."
            ).font(.system(size: 11)).foregroundStyle(.secondary)
          }
          if editor.original?.connected == true {
            Text(
              "This connector has an existing OAuth sign-in. Changing its URL removes that sign-in."
            ).font(.system(size: 12)).foregroundStyle(.secondary)
          }
          if let model, let row = model.rows.first(where: { $0.id == editor.draft.id }) {
            if row.credentialReady == false {
              HStack {
                Text("Saved credentials are locked. Unlock before using this connector.")
                  .foregroundStyle(.secondary)
                Button("Unlock…", systemImage: "lock") {
                  Task { await model.write(.unlock(id: row.id, revision: row.revision)) }
                }.buttonStyle(FlatButtonStyle()).disabled(model.saving)
              }
            }
            WorkspaceRule()
            Text("Sign in").font(.system(size: 13, weight: .semibold))
            if model.signInRow?.id == row.id {
              ConnectorSignInView(model: model, row: row, embedded: true)
            } else {
              HStack {
                if row.connected { Label("Connected", systemImage: "checkmark.circle") }
                Button(row.connected ? "Reconnect…" : "Connect…") {
                  model.error = nil
                  model.signInRow = row
                  Task { await model.beginSignIn() }
                }.buttonStyle(FlatButtonStyle()).disabled(
                  !editor.authorizationEditable || model.saving)
                if row.connected {
                  Button("Disconnect") {
                    Task { await model.write(.disconnect(id: row.id, revision: row.revision)) }
                  }
                  .buttonStyle(QuietButtonStyle()).disabled(
                    !editor.authorizationEditable || model.saving)
                }
              }
              if !editor.authorizationEditable {
                Text("Save URL or credential changes before signing in.").font(.system(size: 11))
                  .foregroundStyle(.secondary)
              }
            }
          }
          WorkspaceRule()
          Text("Available to models").font(.system(size: 13, weight: .semibold))
          Toggle("Every model, including future models", isOn: $editor.scopeAll).toggleStyle(
            .checkbox)
          if !endpoints.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
              ForEach(endpoints) { endpoint in
                Toggle(
                  "\(endpoint.title) · :\(endpoint.port)",
                  isOn: Binding(
                    get: { editor.scopePorts.contains(endpoint.port) },
                    set: {
                      if $0 {
                        editor.scopePorts.insert(endpoint.port)
                      } else {
                        editor.scopePorts.remove(endpoint.port)
                      }
                    })
                )
                .toggleStyle(.checkbox)
              }
            }.disabled(editor.scopeAll)
          }
          if editor.scopeAll || !editor.scopePorts.isEmpty {
            Text("Selected models and their API clients can run tools without per-call approval.")
              .font(.system(size: 11)).foregroundStyle(.secondary)
          }
          WorkspaceRule()
          HStack {
            Button("Check & list tools") { Task { await editor.check() } }.buttonStyle(
              FlatButtonStyle()
            ).disabled(!editor.valid || editor.checking || editor.saving)
            if editor.checking { ProgressView().controlSize(.small) }
          }
          if editor.checked {
            Text(editor.message ?? "Checked").font(.system(size: 12)).foregroundStyle(.secondary)
            Text("\(editor.tools.count) tools discovered").font(.system(size: 12, weight: .medium))
            if !editor.tools.isEmpty {
              DisclosureGroup("Browse \(editor.tools.count) tools") {
                TextField("Search tool names and descriptions", text: $toolQuery).textFieldStyle(
                  .plain)
                LazyVStack(alignment: .leading, spacing: 10) {
                  ForEach(
                    editor.tools.filter {
                      toolQuery.isEmpty || $0.name.localizedCaseInsensitiveContains(toolQuery)
                        || $0.description?.localizedCaseInsensitiveContains(toolQuery) == true
                    }
                  ) { tool in
                    VStack(alignment: .leading, spacing: 4) {
                      Text(tool.name).fontWeight(.medium)
                      Text(tool.description ?? "").foregroundStyle(.secondary)
                    }
                  }
                }
              }.font(.system(size: 11))
            }
          }
          if let error = editor.error {
            Text(error).font(.system(size: 12)).foregroundStyle(PaddockStyle.caution).textSelection(
              .enabled)
          }
        }.padding(22).disabled(editor.saving)
      }.frame(maxHeight: 470)
      WorkspaceRule()
      HStack {
        Spacer()
        Button("Cancel", action: onCancel).buttonStyle(QuietButtonStyle()).disabled(
          editor.saving || model?.busy == true)
        Button(
          editor.saving
            ? "Saving…"
            : editor.checking ? "Checking…" : editor.saveAnyway ? "Save anyway" : "Save connector"
        ) { Task { await editor.save() } }
        .buttonStyle(FlatButtonStyle(primary: true)).disabled(
          !editor.valid || editor.saving || editor.checking || model?.signInRow != nil
            || model?.busy == true)
      }.padding(16)
    }.frame(width: 560).background(PaddockStyle.canvas).presentationBackground(PaddockStyle.canvas)
      .interactiveDismissDisabled()
  }
}

struct IntegrationField<Content: View>: View {
  let title: String
  @ViewBuilder let content: Content
  init(_ title: String, @ViewBuilder content: () -> Content) {
    self.title = title
    self.content = content()
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 7) {
      Text(title).font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
      content.textFieldStyle(.plain).font(.system(size: 13)).padding(10)
        .background(
          PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
        )
        .overlay(
          RoundedRectangle(cornerRadius: PaddockStyle.Radius.control).strokeBorder(
            PaddockStyle.border))
    }
  }
}
