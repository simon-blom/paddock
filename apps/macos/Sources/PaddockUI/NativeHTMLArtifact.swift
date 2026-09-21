import PaddockStudio
import SwiftUI
import WebKit

struct NativeHTMLArtifact: View {
  let html: String
  let origin: URL
  @Environment(\.colorScheme) private var colorScheme
  @State private var session = ArtifactPreviewSession()
  @State private var images = false
  private struct Input: Equatable {
    let html: String
    let origin: URL
    let images: Bool
    let dark: Bool
  }
  private var input: Input {
    .init(html: html, origin: origin, images: images, dark: colorScheme == .dark)
  }
  var body: some View {
    VStack(alignment: .leading, spacing: 0) {
      if !session.blocked.isEmpty {
        Text(
          "\(session.blocked.count) external pictures did not load (\(hosts(session.blocked))) - the preview has no network access."
        ).font(.caption).foregroundStyle(PaddockStyle.caution).padding(10)
        if !images {
          Button("Load pictures") { images = true }
            .buttonStyle(FlatButtonStyle())
            .help(
              "Allows HTTPS pictures for this preview only. Those hosts receive your IP and image requests. Scripts still cannot access the network."
            )
        }
      }
      if !session.failed.isEmpty {
        Text("Pictures could not load: \(hosts(session.failed))").font(.caption).foregroundStyle(
          .secondary)
      }
      ZStack {
        if let webView = session.webView { ArtifactWebSurface(webView: webView) }
        if session.loading { ProgressView().controlSize(.small) }
        if let error = session.error {
          Text(error).foregroundStyle(.secondary).padding(16).textSelection(.enabled)
        }
      }.frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(PaddockStyle.surface)
    }
    .task(id: input) {
      // Coalesce native source edits before starting WebContent work.
      do { try await Task.sleep(for: .milliseconds(200)) } catch { return }
      await session.render(
        html: html, origin: origin, allowImages: images, dark: colorScheme == .dark)
    }
    .onChange(of: html) { _, _ in images = false }
    .onDisappear { session.close() }
  }
  private func hosts(_ values: [String]) -> String {
    let hosts = Array(Set(values.map { URL(string: $0)?.host ?? $0 })).sorted()
    return hosts.prefix(4).map { String($0.prefix(80)) }.joined(separator: ", ")
      + (hosts.count > 4 ? " and \(hosts.count - 4) more" : "")
  }
}

private struct ArtifactWebSurface: NSViewRepresentable {
  let webView: WKWebView
  func makeNSView(context: Context) -> NSView { NSView() }
  func updateNSView(_ container: NSView, context: Context) {
    guard webView.superview !== container else { return }
    for view in container.subviews { view.removeFromSuperview() }
    webView.frame = container.bounds
    webView.autoresizingMask = [.width, .height]
    container.addSubview(webView)
  }
  static func dismantleNSView(_ view: NSView, coordinator: ()) {
    for child in view.subviews { child.removeFromSuperview() }
  }
}
