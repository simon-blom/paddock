import AppKit
import PaddockClient
import SwiftUI

struct ConnectorSignInView: View {
  @Bindable var model: IntegrationsModel
  let row: NativeConnector
  var embedded = false
  @State private var opened = false
  var body: some View {
    VStack(alignment: .leading, spacing: 18) {
      if !embedded {
        Text("Sign in to \(row.label)").font(.system(size: 20, weight: .semibold))
        Text(row.url).font(.system(size: 12)).foregroundStyle(.secondary).textSelection(.enabled)
      }
      if let auth = model.authorization, let url = URL(string: auth.url) {
        Text("Authorization server: \(url.host() ?? "Unknown")").font(
          .system(size: 13, weight: .medium))
        Button(
          opened ? "Reopen browser" : "Continue in browser", systemImage: "arrow.up.right.square"
        ) {
          guard ["https", "http"].contains(url.scheme ?? "") else { return }
          opened = NSWorkspace.shared.open(url)
          if !opened { model.error = "The browser could not be opened." }
        }.buttonStyle(FlatButtonStyle(primary: true))
        if opened { ProgressView("Waiting for sign-in…").controlSize(.small) }
      } else {
        IntegrationField("OAuth client ID - optional") {
          TextField(
            "Pre-registered client ID, if your provider requires one", text: $model.oauthClientID)
        }
        Button(model.busy ? "Discovering authorization…" : "Prepare sign-in") {
          Task { await model.beginSignIn() }
        }.buttonStyle(FlatButtonStyle(primary: true)).disabled(model.busy)
      }
      if let error = model.error {
        Text(error).font(.system(size: 12)).foregroundStyle(PaddockStyle.caution).textSelection(
          .enabled)
      }
      HStack {
        Spacer()
        Button("Cancel sign-in") { Task { await model.cancelSignIn() } }.buttonStyle(
          QuietButtonStyle()
        ).disabled(model.busy)
      }
    }.padding(embedded ? 0 : 24).frame(width: embedded ? nil : 550).background(PaddockStyle.canvas)
      .presentationBackground(
        PaddockStyle.canvas
      ).interactiveDismissDisabled()
      .task(id: model.authorization?.url) {
        if model.authorization != nil { await model.pollSignIn() }
      }
  }
}
