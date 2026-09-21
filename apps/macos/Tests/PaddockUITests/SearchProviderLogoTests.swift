import AppKit
import PaddockStudio
import SwiftUI
import Testing

@testable import PaddockUI

@Suite("Web search provider artwork", .serialized) @MainActor
struct SearchProviderLogoTests {
  @Test func providerChoicesAndArtworkAgree() throws {
    #expect(
      WebSearchForm.providers.dropFirst().map(\.0) == SearchProviderArtwork.allCases.map(\.rawValue)
    )
    #expect(
      WebSearchForm.providers.dropFirst().map(\.1) == SearchProviderArtwork.allCases.map(\.label))
    for provider in SearchProviderArtwork.allCases {
      let image = try #require(ProviderArtwork.image(for: provider.label))
      #expect(image.isValid)
      #expect(abs(image.size.width / image.size.height - provider.aspectRatio) < 0.01)
    }
    #expect(SearchProviderArtwork(rawValue: "") == nil)
    #expect(SearchProviderArtwork(rawValue: "cloud-native-search") == nil)
    #expect(SearchProviderArtwork.exa.rgb(dark: false) == 0x1f40ed)
    #expect(SearchProviderArtwork.exa.rgb(dark: true) == 0x6f88ff)
    #expect(SearchProviderArtwork.tavily.rgb(dark: false) == nil)
  }

  @Test func searchIndicatorUsesTheWebStatePrecedence() {
    for provider in SearchProviderArtwork.allCases {
      #expect(
        SearchCallIndicator(status: "completed", provider: provider.rawValue) == .provider(provider)
      )
      #expect(SearchCallIndicator(status: "failed", provider: provider.rawValue) == .error)
      for running in ["searching", "in_progress"] {
        #expect(SearchCallIndicator(status: running, provider: provider.rawValue) == .progress)
      }
    }
    for unknown in ["", "openrouter", "future-provider"] {
      #expect(SearchCallIndicator(status: "completed", provider: unknown) == .globe)
      #expect(SearchCallIndicator(status: "failed", provider: unknown) == .error)
      #expect(SearchCallIndicator(status: "searching", provider: unknown) == .progress)
    }
  }

  @Test func brandsRenderInColourInBothThemes() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      for provider in SearchProviderArtwork.allCases {
        let host = NSHostingController(
          rootView: SearchProviderLogo(provider: provider, size: 56)
            .padding(12).background(PaddockStyle.surface)
            .environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: CGSize(width: 500, height: 500))
        #expect(abs(size.width - (56 * provider.aspectRatio + 24)) < 1)
        #expect(abs(size.height - 80) < 1)
        let window = window(host, size: size, dark: dark)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(50))
        host.view.layoutSubtreeIfNeeded()
        let bitmap = try capture(host.view)
        // cacheDisplay is Display P3 on this machine. colorAt reports generic
        // RGB; convert the bitmap itself before comparing sRGB channel values.
        let srgb = try #require(bitmap.converting(to: .sRGB, renderingIntent: .default))
        var coloured = 0
        var matching = 0
        let rgb = provider.rgb(dark: dark)
        let expected = rgb ?? 0xfdfaf4  // Tavily's original cream ink, not a white template.
        for y in 0..<srgb.pixelsHigh {
          for x in 0..<srgb.pixelsWide {
            guard let color = srgb.colorAt(x: x, y: y) else { continue }
            let channels = [color.redComponent, color.greenComponent, color.blueComponent]
            if channels.max()! - channels.min()! > 0.15 { coloured += 1 }
            if abs(color.redComponent - CGFloat((expected >> 16) & 255) / 255) < 0.015,
              abs(color.greenComponent - CGFloat((expected >> 8) & 255) / 255) < 0.015,
              abs(color.blueComponent - CGFloat(expected & 255) / 255) < 0.015
            {
              matching += 1
            }
          }
        }
        if rgb != nil { #expect(coloured > 50, "\(provider.label) must not be desaturated") }
        #expect(matching > 50, "\(provider.label) must retain its web brand colour")
        try save(bitmap, name: "logo-\(provider.rawValue)-\(dark)")
      }
    }
  }

  @Test func searchCardsKeepTheCompactHeaderAndExpandOnClick() async throws {
    _ = NSApplication.shared
    for dark in [false, true] {
      for width: CGFloat in [280, 560] {
        let probe = NSView()
        let search = try fixture()
        let host = NSHostingController(
          rootView: VStack(alignment: .leading, spacing: 0) {
            NativeSearchCallView(search: search).background(FrameProbe(view: probe))
            Spacer(minLength: 0)
          }.padding(12).background(PaddockStyle.canvas).environment(
            \.colorScheme, dark ? .dark : .light))
        let window = window(host, size: CGSize(width: width, height: 280), dark: dark)
        defer { window.close() }
        try await Task.sleep(for: .milliseconds(80))
        host.view.layoutSubtreeIfNeeded()
        let collapsed = probe.bounds.height
        #expect(collapsed >= 28 && collapsed <= 36)
        for expand in [true, false, true] {
          let header = probe.convert(
            NSPoint(
              x: probe.bounds.midX,
              y: probe.isFlipped ? 14 : probe.bounds.height - 14), to: nil)
          for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
            window.sendEvent(
              try #require(
                NSEvent.mouseEvent(
                  with: type, location: header,
                  modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                  windowNumber: window.windowNumber, context: nil, eventNumber: 1, clickCount: 1,
                  pressure: 1)))
          }
          try await Task.sleep(for: .milliseconds(80))
          host.view.layoutSubtreeIfNeeded()
          #expect(
            expand ? probe.bounds.height > collapsed + 35 : abs(probe.bounds.height - collapsed) < 1
          )
          #expect(probe.bounds.width <= width - 24)
        }
        try save(capture(host.view), name: "search-expanded-\(Int(width))-\(dark)")
      }
    }
  }

  @Test func sourceHostsAndLinksRemainSafe() {
    #expect(NativeSearchCallView.host("https://www.example.com/page") == "example.com")
    #expect(NativeSearchCallView.safeURL("https://example.com/page") != nil)
    for unsafe in ["javascript:alert(1)", "file:///tmp/a", "https://name:password@example.com"] {
      #expect(NativeSearchCallView.safeURL(unsafe) == nil)
    }
  }

  private func fixture() throws -> StudioState.NativeTranscript.Message.Search {
    try JSONDecoder().decode(
      StudioState.NativeTranscript.Message.Search.self,
      from: Data(
        #"{"id":"search","query":"Swift on macOS","status":"completed","provider":"exa","error":"","sources":[{"title":"Swift documentation","url":"https://www.swift.org/documentation/"},{"title":"","url":"https://developer.apple.com/documentation/swift"}]}"#
          .utf8))
  }

  private func window<Content: View>(_ host: NSHostingController<Content>, size: CGSize, dark: Bool)
    -> NSWindow
  {
    let window = NSWindow(
      contentRect: NSRect(origin: NSPoint(x: -12000, y: -12000), size: size),
      styleMask: [.borderless], backing: .buffered, defer: false)
    host.sizingOptions = []
    window.isReleasedWhenClosed = false
    window.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
    window.contentViewController = host
    window.setContentSize(size)
    window.setFrameOrigin(NSPoint(x: -12000, y: -12000))
    window.orderBack(nil)
    return window
  }

  private func capture(_ view: NSView) throws -> NSBitmapImageRep {
    let bitmap = try #require(view.bitmapImageRepForCachingDisplay(in: view.bounds))
    view.cacheDisplay(in: view.bounds, to: bitmap)
    return bitmap
  }

  private func save(_ bitmap: NSBitmapImageRep, name: String) throws {
    guard let path = ProcessInfo.processInfo.environment["PADDOCK_SEARCH_SNAPSHOTS"] else { return }
    let directory = URL(fileURLWithPath: path, isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try #require(bitmap.representation(using: .png, properties: [:]))
      .write(to: directory.appending(path: name + ".png"))
  }

  private struct FrameProbe: NSViewRepresentable {
    let view: NSView
    func makeNSView(context: Context) -> NSView { view }
    func updateNSView(_ nsView: NSView, context: Context) {}
  }
}
