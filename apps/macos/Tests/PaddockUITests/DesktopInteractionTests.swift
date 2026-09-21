import AppKit
import Foundation
import PaddockClient
import ScreenCaptureKit
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Native OS surfaces", .serialized, .timeLimit(.minutes(2))) @MainActor
struct DesktopInteractionTests {
  @Test func windowSurfacePreservesFrameAndDoesNotCaptureInput() async throws {
    _ = NSApplication.shared
    let view = DesktopWindowBackdropView(frame: NSRect(x: 0, y: 0, width: 530, height: 580))
    view.requestedAppearance = .dark
    let window = NSWindow(
      contentRect: view.frame, styleMask: [.titled, .closable], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = view
    defer { window.close() }
    let frame = window.frame
    for _ in 0..<5 { view.scheduleSurface() }
    try await Task.sleep(for: .milliseconds(50))
    #expect(window.frame == frame)
    #expect(window.styleMask.contains(.fullSizeContentView))
    #expect(window.appearance?.name == .darkAqua)
    #expect(view.hitTest(NSPoint(x: 20, y: 20)) == nil)
    #expect(!view.acceptsFirstResponder)
    view.requestedAppearance = .system
    view.scheduleSurface()
    try await Task.sleep(for: .milliseconds(50))
    #expect(window.appearance == nil)
    #expect(window.frame == frame)
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OS_UI_TEST"] == "1"))
  func quickPanelKeepsDraftFocusAndWorkspaceAcrossReopens() async throws {
    _ = NSApplication.shared
    let model = WorkspaceModel(client: PanelNoCore())
    let controller = QuickQuestionController()
    controller.reduceMotion = { true }
    controller.draft.text = "Synthetic draft - never sent"
    controller.show(workspace: model) { _ in }
    let panel = try #require(controller.panel)
    defer { panel.orderOut(nil) }
    let web = model.chat.webView
    try await Task.sleep(for: .milliseconds(300))
    #expect(panel.isVisible)
    #expect(panel.firstResponder is DraftTextView)
    if let editor = panel.firstResponder as? DraftTextView {
      #expect(editor.frame.width <= editor.enclosingScrollView!.contentView.bounds.width + 1)
      #expect(editor.font?.pointSize == StudioDraftEditor.textFontSize)
      #expect(editor.textContainer?.lineFragmentPadding == StudioDraftEditor.horizontalTextPadding)
      #expect(editor.textContainerInset.height == StudioDraftEditor.verticalTextPadding)
    }
    #expect(panel.level == .floating)
    #expect(!panel.isOpaque)  // Transparent corners, opaque content.
    #expect(!panel.styleMask.contains(.titled))
    #expect(!panel.styleMask.contains(.fullSizeContentView))
    #expect(panel.contentView!.fittingSize.width <= 462)
    #expect(panel.contentView!.fittingSize.height <= panel.contentView!.bounds.height + 1)
    panel.performClose(nil)
    #expect(!panel.isVisible)
    #expect(controller.draft.hasContent)
    controller.show(workspace: model) { _ in }
    for _ in 0..<5 {
      controller.hide()
      #expect(!panel.isVisible)
      controller.show(workspace: model) { _ in }
      #expect(controller.draft.text == "Synthetic draft - never sent")
      #expect(model.chat.webView === web)
      #expect(
        NSApp.windows.filter { $0.identifier?.rawValue == "quick-question" && $0.isVisible }.count
          == 1)
    }
    for dark in [false, true] {
      panel.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      try await Task.sleep(for: .milliseconds(100))
      if let output = ProcessInfo.processInfo.environment["PADDOCK_UI_SNAPSHOT_DIR"],
        let view = panel.contentView,
        let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds)
      {
        view.cacheDisplay(in: view.bounds, to: bitmap)
        let directory = URL(fileURLWithPath: output, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bitmap.representation(using: .png, properties: [:])?.write(
          to: directory.appending(path: "quick-question-\(dark ? "dark" : "light").png"))
      }
    }
    controller.hide()
    await model.shutdown()
  }

  // cacheDisplay omits the title bar and cannot catch wallpaper-tinted window
  // chrome. Capture the presented window over vivid backdrops instead.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OS_UI_TEST"] == "1"))
  func presentedQuickQuestionHasNeutralSurfaceAndEdges() async throws {
    guard #available(macOS 15.2, *) else { return }
    _ = NSApplication.shared
    let defaults = UserDefaults.standard
    let oldAppearance = defaults.object(forKey: "workspaceAppearance")
    defer { defaults.set(oldAppearance, forKey: "workspaceAppearance") }
    let model = WorkspaceModel(client: PanelNoCore())
    let controller = QuickQuestionController()
    controller.reduceMotion = { true }
    controller.show(workspace: model) { _ in }
    let panel = try #require(controller.panel)
    defer { controller.hide() }
    try await checkPresentedSurface(panel, name: "quick-question") { appearance in
      defaults.set(appearance.rawValue, forKey: "workspaceAppearance")
    }
    #expect(panel.firstResponder is DraftTextView)
    await model.shutdown()
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OS_UI_TEST"] == "1"))
  func presentedSettingsFormHasNeutralTitlebarAndEdges() async throws {
    guard #available(macOS 15.2, *) else { return }
    _ = NSApplication.shared
    let domain = "io.truespar.paddock.surface-test.\(UUID())"
    let defaults = try #require(UserDefaults(suiteName: domain))
    defer { defaults.removePersistentDomain(forName: domain) }
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 530, height: 580),
      styleMask: [.titled, .closable], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.title = "Settings - surface checks"
    // Real Settings surface without login/notification services: those require
    // the signed app, not swiftpm's unbundled testing helper. Do not request OS permissions.
    window.contentViewController = NSHostingController(
      rootView: DesktopSettingsSurface {
        Section("System integration") {
          Toggle("Synthetic enabled setting", isOn: .constant(true))
          Text("Synthetic status").font(.caption).foregroundStyle(.secondary)
        }.listRowBackground(PaddockStyle.surface)
      }.defaultAppStorage(defaults))
    window.center()
    window.makeKeyAndOrderFront(nil)
    defer { window.close() }
    try await checkPresentedSurface(window, name: "settings") { appearance in
      defaults.set(appearance.rawValue, forKey: "workspaceAppearance")
    }
  }

  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_OS_UI_TEST"] == "1"))
  func presentedConfirmationHasNeutralEdges() async throws {
    guard #available(macOS 15.2, *) else { return }
    _ = NSApplication.shared
    let alert = DesktopAlert.make()
    alert.messageText = "Synthetic confirmation"
    alert.informativeText = "No model operation or destructive action is performed."
    alert.addButton(withTitle: "Cancel")
    alert.addButton(withTitle: "Continue")
    alert.layout()
    let keys = alert.buttons.map(\.keyEquivalent)
    let window = alert.window
    window.center()
    window.orderFrontRegardless()
    defer { window.orderOut(nil) }
    try await checkPresentedSurface(window, name: "confirmation") { appearance in
      switch appearance {
      case .system: window.appearance = nil
      case .dark: window.appearance = NSAppearance(named: .darkAqua)
      case .light: window.appearance = NSAppearance(named: .aqua)
      }
    }
    #expect(alert.buttons.map(\.keyEquivalent) == keys)
    #expect(alert.buttons.map(\.title) == ["Cancel", "Continue"])
  }

  @Test func asynchronousDropHoldsAdmissionUntilResolved() async throws {
    let model = WorkspaceModel(client: PanelNoCore())
    let quick = QuickQuestionModel()
    quick.text = "Synthetic pending file"
    let provider = NSItemProvider()
    provider.registerDataRepresentation(forTypeIdentifier: "public.file-url", visibility: .all) {
      completion in
      DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) {
        completion(Data("file:///tmp/paddock-synthetic-not-opened.pdf".utf8), nil)
      }
      return nil
    }
    #expect(quick.addDroppedFiles([provider]))
    #expect(quick.loadingDrops)
    #expect(
      !(await quick.handoff(to: model) { Issue.record("Must not open before the file arrives") }))
    for _ in 0..<100 {
      if !quick.loadingDrops { break }
      try await Task.sleep(for: .milliseconds(10))
    }
    #expect(!quick.loadingDrops)
    #expect(quick.attachments.count == 1)
    #expect(quick.text == "Synthetic pending file")
    await model.shutdown()
  }
  @Test func lateProviderCallbackCannotResurrectATimedOutDrop() async throws {
    let provider = NSItemProvider()
    provider.registerDataRepresentation(forTypeIdentifier: "public.file-url", visibility: .all) {
      completion in
      DispatchQueue.global().asyncAfter(deadline: .now() + 0.15) {
        completion(Data("file:///tmp/paddock-late-not-opened.pdf".utf8), nil)
      }
      return nil
    }
    do {
      _ = try await QuickDropLoad().load(provider, deadline: .now.advanced(by: .milliseconds(20)))
      Issue.record("A late drop was accepted")
    } catch { #expect(error.localizedDescription.contains("timed out")) }
    try await Task.sleep(for: .milliseconds(200))
  }
}

@MainActor @available(macOS 15.2, *)
private func checkPresentedSurface(
  _ window: NSWindow, name: String, setAppearance: (WorkspaceAppearance) -> Void
) async throws {
  let backdrop = NSWindow(
    contentRect: window.frame.insetBy(dx: -40, dy: -40),
    styleMask: [.borderless], backing: .buffered, defer: false)
  backdrop.isReleasedWhenClosed = false
  let oldLevel = window.level
  // A running preview may have its own floating Quick Question panel. Keep
  // the entire fixture above that band; checking the backdrop alone cannot
  // detect a foreign panel occluding the middle of our window.
  let fixtureLevel = max(oldLevel.rawValue, NSWindow.Level.floating.rawValue) + 10
  backdrop.level = NSWindow.Level(rawValue: fixtureLevel)
  window.level = NSWindow.Level(rawValue: fixtureLevel + 1)
  backdrop.isOpaque = true
  backdrop.orderFrontRegardless()
  defer {
    window.level = oldLevel
    backdrop.close()
  }
  let oldAppearance = NSApp.appearance
  defer { NSApp.appearance = oldAppearance }
  for (appearance, dark): (WorkspaceAppearance, Bool) in [
    (.dark, true), (.light, false), (.system, true), (.system, false),
  ] {
    NSApp.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
    setAppearance(appearance)
    try await Task.sleep(for: .milliseconds(200))
    for green in [false, true] {
      backdrop.setFrame(window.frame.insetBy(dx: -40, dy: -40), display: true)
      backdrop.backgroundColor = green ? .green : .blue
      backdrop.orderFrontRegardless()
      window.orderFrontRegardless()
      try await Task.sleep(for: .milliseconds(200))
      let screenTop = try #require(NSScreen.screens.first?.frame.maxY)
      try #require(window.frame.width > 200 && window.frame.height > 100)
      let frame = window.frame.insetBy(dx: -8, dy: -8)
      let image = try await SCScreenshotManager.captureImage(
        in: CGRect(
          x: frame.minX, y: screenTop - frame.maxY, width: frame.width, height: frame.height))
      let bitmap = NSBitmapImageRep(cgImage: image)
      if let output = ProcessInfo.processInfo.environment["PADDOCK_UI_SNAPSHOT_DIR"] {
        let directory = URL(fileURLWithPath: output, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try #require(bitmap.representation(using: .png, properties: [:])).write(
          to: directory.appending(
            path: "presented-\(name)-\(appearance.rawValue)-\(dark)-\(green).png"))
      }
      let context = try #require(
        CGContext(
          data: nil, width: image.width, height: image.height, bitsPerComponent: 8,
          bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
      context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
      let pixels = try #require(context.data?.assumingMemoryBound(to: UInt8.self))
      // Points are top-left relative to the actual window, not its content view.
      func sample(_ x: CGFloat, _ y: CGFloat) -> [CGFloat] {
        let offset =
          Int((y + 8) / frame.height * CGFloat(image.height)) * context.bytesPerRow
          + Int((x + 8) / frame.width * CGFloat(image.width)) * 4
        return (0..<3).map { CGFloat(pixels[offset + $0]) / 255 }
      }
      let behind = sample(-5, window.frame.height / 2)
      #expect(
        green ? behind[1] > behind[2] + 0.2 : behind[2] > behind[1] + 0.2,
        "Backdrop not visible: \(behind)")
      let points: [(CGFloat, CGFloat)] = [
        (window.frame.width * 0.25, 10),  // title bar, away from title/traffic lights
        (5, window.frame.height / 2), (window.frame.width - 5, window.frame.height / 2),
        (window.frame.width / 2, window.frame.height - 5),
      ]
      let expected = PaddockStyle.nsColor("canvas", dark: dark).redComponent
      for (x, y) in points {
        let rgb = sample(x, y)
        #expect(
          rgb.allSatisfy { abs($0 - expected) < 0.03 },
          "\(name) \(appearance.rawValue) dark=\(dark), green=\(green), edge=(\(x),\(y)): \(rgb)")
      }
    }
  }
}
private struct PanelNoCore: ManagerLoading {
  func snapshot() async throws -> ManagerSnapshot {
    throw ManagerError.core("Synthetic offline manager")
  }
}
