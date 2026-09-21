import PaddockClient
import SwiftUI

struct ConnectionReviewView: View {
  @Bindable var editor: ConnectionEditor
  let onCancel: () -> Void
  @State private var search = ""
  @FocusState private var focused: Field?
  private enum Field { case name, key }
  private var frozen: Bool { editor.checking || editor.saving || editor.checked }
  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      HStack(spacing: 12) {
        if let vendor = editor.service.vendor { ModelAvatar(vendor: vendor, size: 30) }
        VStack(alignment: .leading, spacing: 4) {
          Text(
            editor.service != .custom
              ? "\(editor.original?.hasKey == true ? "Replace" : "Add") \(editor.service.rawValue) key"
              : (editor.original == nil ? "Add endpoint" : "Edit endpoint")
          )
          .font(.system(size: 21, weight: .semibold))
        }
      }
      PaddockScrollView {
        VStack(alignment: .leading, spacing: 18) {
          VStack(alignment: .leading, spacing: 14) {
            if editor.service == .custom {
              field("Name") {
                TextField("Connection name", text: $editor.draft.name).focused(
                  $focused, equals: .name
                )
                .accessibilityIdentifier("connection-name")
              }
              field("API format") {
                Dropdown(title: "API format", value: formatName, fillsWidth: true) {
                  Picker("API format", selection: $editor.draft.kind) {
                    Text("Chat Completions compatible").tag("openai-compat")
                    Text("OpenAI Responses").tag("openai")
                    Text("Anthropic Messages").tag("anthropic")
                  }.pickerStyle(.inline)
                }
              }
              field("API base URL") {
                TextField("https://example.com/v1", text: $editor.draft.baseUrl)
                  .accessibilityIdentifier("connection-base-url")
                  .help(
                    "HTTPS is required except for localhost. Use the base URL, not /chat/completions."
                  )
              }
              Toggle("This endpoint requires no API key", isOn: $editor.noAuthentication)
                .toggleStyle(.checkbox).font(.system(size: 12))
                .accessibilityIdentifier("connection-no-auth")
            }
            if !editor.noAuthentication {
              field("API key") {
                SecureField(
                  editor.original?.hasKey == true
                    ? "******" : "Paste your API key", text: $editor.key
                )
                .focused($focused, equals: .key).accessibilityIdentifier("connection-key")
                .accessibilityHint(
                  editor.original?.hasKey == true
                    ? "A key is saved. Enter a new key to replace it." : "Enter an API key.")
              }
            }
            if editor.service != .custom {
              DisclosureGroup("Account name · \(editor.draft.name)") {
                field("Name") {
                  TextField("Account name", text: $editor.draft.name)
                    .accessibilityIdentifier("connection-name")
                }.padding(.top, 8)
              }.font(.system(size: 11)).foregroundStyle(.secondary)
            }
          }.disabled(frozen)
          if editor.service == .custom && (editor.checked || !editor.models.isEmpty) {
            WorkspaceRule()
            modelChoices
          }
          if let error = editor.error {
            Label(error, systemImage: "exclamationmark.circle")
              .font(.system(size: 12)).foregroundStyle(PaddockStyle.caution)
              .fixedSize(horizontal: false, vertical: true).textSelection(.enabled)
              .accessibilityIdentifier("connection-error")
          } else if editor.checked {
            Label(editor.job?.message ?? "Connection checked.", systemImage: "checkmark.circle")
              .font(.system(size: 12)).foregroundStyle(.secondary)
              .fixedSize(horizontal: false, vertical: true)
              .accessibilityIdentifier("connection-checked")
          }
        }.padding(.trailing, 2)
      }.frame(maxHeight: 420)
      WorkspaceRule()
      HStack(spacing: 12) {
        Button("Cancel", action: onCancel).buttonStyle(FlatButtonStyle())
          .disabled(editor.saving).keyboardShortcut(.cancelAction)
        if editor.checked {
          Button("Edit details") { Task { await editor.editDetails() } }
            .buttonStyle(QuietButtonStyle()).disabled(editor.saving)
        }
        Spacer()
        if editor.checking || editor.saving { ProgressView().controlSize(.small) }
        if editor.checked || editor.saving {
          Button(editor.saving ? "Saving…" : "Save connection") { Task { await editor.save() } }
            .buttonStyle(FlatButtonStyle(primary: true)).disabled(editor.saving)
            .keyboardShortcut(.defaultAction).accessibilityIdentifier("connection-save")
        } else {
          Button(editor.checking ? "Checking…" : "Check connection") {
            Task { await editor.check() }
          }
          .buttonStyle(FlatButtonStyle(primary: true)).disabled(!editor.canCheck)
          .keyboardShortcut(.defaultAction).accessibilityIdentifier("connection-check")
        }
      }
    }.padding(26).frame(width: 520).background(PaddockStyle.canvas)
      .presentationBackground(PaddockStyle.canvas).interactiveDismissDisabled()
      .task { focused = editor.service == .custom ? .name : .key }
  }
  private var formatName: String {
    switch editor.draft.kind {
    case "openai": "OpenAI Responses"
    case "anthropic": "Anthropic Messages"
    default: "Chat Completions compatible"
    }
  }
  private func field<Content: View>(_ title: String, @ViewBuilder content: () -> Content)
    -> some View
  {
    VStack(alignment: .leading, spacing: 7) {
      Text(title).font(.system(size: 11, weight: .medium)).foregroundStyle(.secondary)
      content().textFieldStyle(.plain).font(.system(size: 13)).padding(10)
        .background(
          PaddockStyle.surface, in: RoundedRectangle(cornerRadius: PaddockStyle.Radius.control)
        )
        .overlay(
          RoundedRectangle(cornerRadius: PaddockStyle.Radius.control).strokeBorder(
            PaddockStyle.border))
    }
  }
  private var modelChoices: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("Models in Studio").font(.system(size: 12, weight: .medium))
        Spacer()
        Text("\(editor.models.count) selected").font(.system(size: 11)).foregroundStyle(.secondary)
      }
      if !editor.discovered.isEmpty {
        TextField("Find a model", text: $search).textFieldStyle(.plain).font(.system(size: 12))
          .accessibilityIdentifier("connection-model-search")
      }
      PaddockScrollView {
        LazyVStack(alignment: .leading, spacing: 0) {
          // Preserve previously saved models even if a provider stops listing
          // them. Removing one is explicit; catalog drift cannot erase a pick.
          ForEach(
            editor.models.filter { pick in !editor.discovered.contains { $0.id == pick.id } },
            id: \.pickKey
          ) { pick in
            HStack {
              VStack(alignment: .leading, spacing: 3) {
                Text(pick.display ?? pick.id).lineLimit(1)
                Text(pick.provider.map { "Provider: \($0) · no fallback" } ?? pick.id)
                  .font(.system(size: 10)).foregroundStyle(.secondary).lineLimit(1)
              }
              Spacer()
              Button("Remove from Studio", systemImage: "minus.circle") {
                editor.models.removeAll { $0.pickKey == pick.pickKey }
              }.labelStyle(.iconOnly).buttonStyle(QuietButtonStyle()).disabled(editor.saving)
            }.padding(.vertical, 7)
          }
          ForEach(
            editor.discovered.filter {
              search.isEmpty || $0.id.localizedCaseInsensitiveContains(search)
                || $0.display?.localizedCaseInsensitiveContains(search) == true
            }
          ) { model in
            Button {
              editor.toggle(model)
            } label: {
              HStack(spacing: 10) {
                Image(
                  systemName: editor.models.contains { $0.id == model.id && $0.provider == nil }
                    ? "checkmark.square.fill" : "square")
                Text(model.display ?? model.id).lineLimit(1)
                Spacer(minLength: 0)
              }.frame(height: 32).contentShape(Rectangle())
            }.buttonStyle(QuietButtonStyle()).disabled(editor.saving)
          }
        }.font(.system(size: 12))
      }.frame(
        height: min(156, max(36, CGFloat(editor.discovered.count + editor.models.count) * 36)))
      if editor.models.isEmpty {
        Text(
          editor.openRouter
            ? "Save the account, then choose models from the catalog."
            : "Select models to make them available in the composer."
        )
        .font(.system(size: 11)).foregroundStyle(.secondary)
      }
    }
  }
}
