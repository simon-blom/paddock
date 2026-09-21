import AppKit
import ScreenCaptureKit
import SwiftUI
import Testing

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native popover surfaces", .serialized, .timeLimit(.minutes(1))) @MainActor
struct StudioPopoverTests {
  @Test func extentIsInertOutsideAPopover() async {
    _ = NSApplication.shared
    let view = StudioPopoverExtentView(frame: NSRect(x: 0, y: 0, width: 120, height: 80))
    let window = NSWindow(
      contentRect: view.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = view
    defer { window.close() }
    let frame = window.frame
    view.scheduleUpdate()
    try? await Task.sleep(for: .milliseconds(50))
    #expect(window.frame == frame)
    #expect(view.hitTest(NSPoint(x: 10, y: 10)) == nil)
    #expect(!view.acceptsFirstResponder)
  }

  @Test func extentHandlesALatePopoverResponderWithoutPolling() async {
    _ = NSApplication.shared
    let view = StudioPopoverExtentView(frame: NSRect(x: 0, y: 0, width: 120, height: 80))
    let window = NSWindow(
      contentRect: view.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = view
    defer { window.close() }
    try? await Task.sleep(for: .milliseconds(50))
    let popover = NSPopover()
    popover.hasFullSizeContent = false
    let previous = view.nextResponder
    view.nextResponder = popover
    defer { view.nextResponder = previous }
    view.needsLayout = true
    view.layoutSubtreeIfNeeded()
    try? await Task.sleep(for: .milliseconds(50))
    #expect(popover.hasFullSizeContent)
    #expect(!window.isVisible)
    #expect(!popover.isShown)
  }

  // This gate needs an unlocked desktop and screen-capture access. Ordinary
  // cacheDisplay snapshots omit NSPopover's material/arrow and masked strips,
  // so they cannot detect the bug even when the content looks perfectly flat.
  @Test(.enabled(if: ProcessInfo.processInfo.environment["PADDOCK_NATIVE_POPOVER_TEST"] == "1"))
  func presentedPagesAreOpaqueAtEveryEdgeAndKeepNativeBehavior() async throws {
    guard #available(macOS 15.2, *) else { return }
    _ = NSApplication.shared
    let oldPolicy = NSApp.activationPolicy()
    NSApp.setActivationPolicy(.regular)
    defer { NSApp.setActivationPolicy(oldPolicy) }
    let state = PopoverFixtureState()
    let controller = NSHostingController(rootView: PopoverFixture(state: state))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 760, height: 850),
      styleMask: [.titled, .closable], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.title = "Popover checks - synthetic PDF"
    // The product's floating Quick Question panel can be open during checks.
    // Keep synthetic captures above it without changing the product's ordering.
    window.level = NSWindow.Level(rawValue: NSWindow.Level.floating.rawValue + 10)
    window.contentViewController = controller
    window.center()
    window.makeKeyAndOrderFront(nil)
    NSApp.activate(ignoringOtherApps: true)
    try await Task.sleep(for: .milliseconds(150))
    window.orderFrontRegardless()
    defer {
      state.shown = false
      window.close()
    }
    for dark in [true, false, true] {
      state.dark = dark
      window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
      state.shown = true
      let popover = try await presentedPopover()
      try await Task.sleep(for: .milliseconds(200))
      #expect(popover.hasFullSizeContent)
      let content = try #require(popover.contentViewController?.view)
      let popupWindow = try #require(content.window)
      let popupLevel = popupWindow.level
      popupWindow.level = NSWindow.Level(rawValue: window.level.rawValue + 1)
      defer { popupWindow.level = popupLevel }
      #expect(abs(content.frame.width - popupWindow.frame.width) < 1)
      #expect(abs(content.frame.height - popupWindow.frame.height) < 1)
      let fields = descendants(content).compactMap { $0 as? NSTextField }
        .filter { $0.isEditable }
      #expect(fields.count == 2)
      for field in fields {
        let frame = field.convert(field.bounds, to: content)
        #expect(content.safeAreaLayoutGuide.frame.contains(frame))
      }
      // Change the background under the same open popover. Each edge must
      // stay on the popup palette, not follow blue/green material samples.
      for green in [false, true] {
        state.green = green
        try await Task.sleep(for: .milliseconds(160))
        let image = try await capture(popupWindow)
        try checkEdges(image, view: content, dark: dark, green: green)
        if let directory = ProcessInfo.processInfo.environment["PADDOCK_UI_SNAPSHOT_DIR"] {
          let url = URL(fileURLWithPath: directory, isDirectory: true)
          try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
          let png = try #require(image.representation(using: .png, properties: [:]))
          try png.write(
            to: url.appending(path: "presented-pages-\(dark)-\(green).png"))
        }
      }
      // Real field editing still updates the per-file binding after the
      // background expands. No simulated send, file I/O or inference.
      let first = try #require(fields.first)
      first.stringValue = "2"
      NotificationCenter.default.post(name: NSControl.textDidChangeNotification, object: first)
      try await Task.sleep(for: .milliseconds(60))
      #expect(state.file.from == 2 || state.file.to == 2)
      popover.performClose(nil)
      for _ in 0..<50 {
        if !popover.isShown && !state.shown { break }
        try await Task.sleep(for: .milliseconds(20))
      }
      #expect(!state.shown)
      #expect(!popover.isShown)
    }
  }

  private func presentedPopover() async throws -> NSPopover {
    for _ in 0..<100 {
      for window in NSApp.windows where window.isVisible {
        var responder: NSResponder? = window.contentView
        var seen = Set<ObjectIdentifier>()
        while let current = responder, seen.insert(ObjectIdentifier(current)).inserted {
          if let popover = current as? NSPopover, popover.isShown { return popover }
          responder = current.nextResponder
        }
      }
      try await Task.sleep(for: .milliseconds(20))
    }
    throw PopoverTestFailure(message: "Native popover did not appear")
  }

  @available(macOS 15.2, *)
  private func capture(_ window: NSWindow) async throws -> NSBitmapImageRep {
    try #require(window.isVisible)
    let screenTop = try #require(NSScreen.screens.first?.frame.maxY)
    let frame = CGRect(
      x: window.frame.minX, y: screenTop - window.frame.maxY,
      width: window.frame.width, height: window.frame.height)
    return NSBitmapImageRep(cgImage: try await SCScreenshotManager.captureImage(in: frame))
  }

  private func checkEdges(_ image: NSBitmapImageRep, view: NSView, dark: Bool, green: Bool) throws {
    // Normalize the capture into explicit sRGB bytes. colorAt()/device RGB
    // otherwise applies the display's generic-RGB gamma a second time.
    let cgImage = try #require(image.cgImage)
    let context = try #require(
      CGContext(
        data: nil, width: cgImage.width, height: cgImage.height, bitsPerComponent: 8,
        bytesPerRow: cgImage.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: cgImage.width, height: cgImage.height))
    let pixels = try #require(context.data?.assumingMemoryBound(to: UInt8.self))
    let safe = view.safeAreaLayoutGuide.frame
    #expect(view.safeAreaInsets.top > 0 || view.safeAreaInsets.bottom > 0)
    let points = [
      NSPoint(x: safe.midX, y: safe.maxY - 3),
      NSPoint(x: safe.midX, y: safe.minY + 3),
      NSPoint(x: safe.minX + 3, y: safe.midY),
      NSPoint(x: safe.maxX - 3, y: safe.midY),
    ]
    let expected = PaddockStyle.nsColor("popup", dark: dark)
    func sample(_ point: NSPoint) throws -> [CGFloat] {
      let windowPoint = view.convert(point, to: nil)
      let size = try #require(view.window?.frame.size)
      let x = Int(windowPoint.x / size.width * CGFloat(image.pixelsWide))
      let y = Int((size.height - windowPoint.y) / size.height * CGFloat(image.pixelsHigh))
      try #require(x >= 0 && x < image.pixelsWide && y >= 0 && y < image.pixelsHigh)
      let offset = y * context.bytesPerRow + x * 4
      return (0..<3).map { CGFloat(pixels[offset + $0]) / 255 }
    }
    // Verify that the intended vivid test window is actually behind us. An
    // occluded/off-screen fixture must not produce a false "no tint" pass.
    let backdrop = try sample(NSPoint(x: 1, y: safe.midY))
    #expect(
      green ? backdrop[1] > backdrop[2] + 0.2 : backdrop[2] > backdrop[1] + 0.2,
      "Synthetic backdrop was not visible: \(backdrop)")
    for point in points {
      let color = try sample(point)
      #expect(abs(color[0] - expected.redComponent) < 0.03, "Edge \(point): \(color)")
      #expect(abs(color[1] - expected.greenComponent) < 0.03, "Edge \(point): \(color)")
      #expect(abs(color[2] - expected.blueComponent) < 0.03, "Edge \(point): \(color)")
    }
    // AppKit may put the arrow above or below to keep the popover on-screen.
    // The centered unsafe-area sample on the arrow side must also be opaque;
    // the opposite sample is intentionally outside the clipped native shape.
    let arrowSamples = try [safe.minY - 3, safe.maxY + 3].map {
      try sample(NSPoint(x: safe.midX, y: $0))
    }
    #expect(
      arrowSamples.contains {
        abs($0[0] - expected.redComponent) < 0.03
          && abs($0[1] - expected.greenComponent) < 0.03
          && abs($0[2] - expected.blueComponent) < 0.03
      }, "Neither arrow edge matches the popup palette: \(arrowSamples)")
  }

  private func descendants(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap { descendants($0) }
  }
}

private struct PopoverTestFailure: Error { let message: String }

@MainActor @Observable private final class PopoverFixtureState {
  var shown = false
  var dark = true
  var green = false
  var file: StudioAttachment = {
    var value = StudioAttachment(
      id: "synthetic-pdf", name: "Synthetic report.pdf", mime: "application/pdf", size: 8192,
      phase: "Ready")
    value.pages = 12
    return value
  }()
}

private struct PopoverFixture: View {
  @Bindable var state: PopoverFixtureState
  var body: some View {
    VStack {
      Spacer()
      Button("PDF pages") { state.shown = true }
        .popover(isPresented: $state.shown, arrowEdge: .top) {
          StudioAttachmentOptions(
            attachment: $state.file, canRasterPDF: true, onDone: { state.shown = false })
        }
      Spacer().frame(height: 100)
    }.frame(width: 760, height: 850)
      .background(state.green ? Color.green : Color.blue)
      .preferredColorScheme(state.dark ? .dark : .light)
  }
}
