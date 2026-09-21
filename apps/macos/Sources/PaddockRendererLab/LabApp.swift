import AppKit
import PaddockRendererHost
import SwiftUI

@main
enum RendererLabApp {
  @MainActor static func main() {
    let app = NSApplication.shared
    let lifecycle = LabLifecycle()
    app.delegate = lifecycle
    app.setActivationPolicy(.regular)
    withExtendedLifetime(lifecycle) { app.run() }
  }
}

@MainActor
final class LabLifecycle: NSObject, NSApplicationDelegate {
  private var window: NSWindow?

  func applicationDidFinishLaunching(_ notification: Notification) {
    let args = ProcessInfo.processInfo.arguments
    func argument(_ key: String) -> String? {
      guard let index = args.firstIndex(of: key), args.indices.contains(index + 1) else {
        return nil
      }
      return args[index + 1]
    }
    let assets =
      argument("--assets").map { URL(fileURLWithPath: $0) }
      ?? Bundle.main.resourceURL!.appendingPathComponent("RendererLab")
    let output =
      argument("--output").map { URL(fileURLWithPath: $0) }
      ?? FileManager.default.temporaryDirectory.appendingPathComponent(
        "paddock-renderer-lab-\(UUID().uuidString)")
    print("Rendering Lab report: \(output.path)")
    // An explicit, non-restored diagnostic window also works when invoked as a
    // SwiftPM executable. The product app's SwiftUI scene lifecycle is untouched.
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 1200, height: 850),
      styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false
    )
    window.title = "Paddock Rendering Lab"
    window.isReleasedWhenClosed = false
    let content = NSHostingView(
      rootView: LabContent(
        assets: assets, output: output,
        autorun: argument("--run").flatMap(LabCommand.init(rawValue:))
          ?? (args.contains("--run-all") ? .all : nil),
        automaticSnapshots: !args.contains("--no-snapshots"),
        externalGuard: args.contains("--external-guard")))
    content.sizingOptions = [.minSize]
    window.contentView = content
    window.setContentSize(NSSize(width: 1200, height: 850))
    window.center()
    self.window = window
    let menu = NSMenu()
    let appMenu = NSMenu()
    appMenu.addItem(
      withTitle: "Quit Rendering Lab", action: #selector(NSApplication.terminate(_:)),
      keyEquivalent: "q")
    let appItem = NSMenuItem()
    appItem.submenu = appMenu
    menu.addItem(appItem)
    let edit = NSMenu(title: "Edit")
    for (title, action, key) in [("Copy", "copy:", "c"), ("Select All", "selectAll:", "a")] {
      edit.addItem(withTitle: title, action: Selector(action), keyEquivalent: key)
    }
    let editItem = NSMenuItem()
    editItem.submenu = edit
    menu.addItem(editItem)
    NSApp.mainMenu = menu
    window.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
  }

  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

struct LabContent: View {
  let assets: URL
  let output: URL
  let autorun: LabCommand?
  let automaticSnapshots: Bool
  let externalGuard: Bool
  @State private var session: LabSession?
  @State private var query = ""

  var body: some View {
    Group {
      if let session {
        VStack(spacing: 0) {
          HStack(spacing: 12) {
            Text("Rendering Lab").font(.headline)
            Text(session.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            Spacer()
            Button("Run all") { session.command(.all) }.disabled(!session.ready || session.running)
            Button("Scale tests") { session.command(.scale) }.disabled(
              !session.ready || session.running)
            Button("Appearance") { session.command(.theme) }.disabled(!session.ready)
            Button("Snapshot") { session.snapshot() }.disabled(!session.ready)
            Button("Reload") { session.reload() }
          }.padding(14)
          HStack {
            TextField("Find in rendered text", text: $query).onSubmit { session.find(query) }
            Button("Find") { session.find(query) }.disabled(query.isEmpty)
            Text("PDF uses its own search fixture; canvas text needs the engine's selection layer.")
              .font(.caption).foregroundStyle(.secondary)
          }.padding(.horizontal, 14).padding(.bottom, 10)
          Divider()
          LabWebView(session: session)
        }
      } else {
        ProgressView("Preparing isolated renderer…")
      }
    }
    .frame(minWidth: 900, minHeight: 680)
    .onAppear {
      // Mount the native host before starting WebKit and the visual tests.
      if session == nil {
        let layerCapture =
          externalGuard && ProcessInfo.processInfo.arguments.contains("--trace-layers")
        session = LabSession(
          assetsRoot: assets, output: output, autorun: layerCapture ? nil : autorun,
          automaticSnapshots: automaticSnapshots, externalGuard: externalGuard)
        if layerCapture, let session, let autorun {
          Task { await LayerCapture.run(session: session, output: output, command: autorun) }
        }
        if externalGuard, ProcessInfo.processInfo.arguments.contains("--heap-snapshot"), let session
        {
          Task { await HeapCapture.run(session: session, output: output) }
        }
      }
    }
  }
}

struct LabWebView: NSViewRepresentable {
  let session: LabSession
  func makeNSView(context: Context) -> NSView { session.webView }
  func updateNSView(_ view: NSView, context: Context) {}
}
