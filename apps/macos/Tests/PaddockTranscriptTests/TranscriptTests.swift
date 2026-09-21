import AppKit
import Foundation
import PaddockClient
import PaddockWebAssets
import Testing
import WebKit

@testable import PaddockTranscript

@Suite("Production transcript boundary")
struct TranscriptTests {
  @Test func resourcesAndPolicyExcludePrivilegedOrigins() {
    let assets = BundledAssets(
      root: URL(fileURLWithPath: "/tmp/transcript-assets"), surface: .transcript)
    #expect(assets.resource(for: URL(string: "paddock-render://bundle/assets/a.js")!) != nil)
    for address in [
      "file:///etc/passwd", "https://example.com/a.js", "paddock-lab://bundle/a.js",
      "paddock-render://bundle/%2e%2e/a.js", "paddock-render://user@bundle/a.js",
      "paddock-render://bundle/a.key",
    ] {
      #expect(assets.resource(for: URL(string: address)!) == nil)
    }
    #expect(assets.contentSecurityPolicy.contains("frame-src 'none'"))
    #expect(!assets.contentSecurityPolicy.contains("https:"))
    #expect(!assets.contentSecurityPolicy.contains("'unsafe-eval'"))
  }
}
