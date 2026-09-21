import Foundation
import PaddockClient
import Testing

@testable import PaddockConversationCore

@Suite("Native conversation transport security")
struct TransportPolicyTests {
  @Test func onlyExactLoopbackDescriptorIsAccepted() throws {
    let token = String(repeating: "a", count: 64)
    for origin in [
      "https://example.com", "http://localhost:1234", "http://127.0.0.1:0",
      "http://user:password@127.0.0.1:1234", "http://127.0.0.1:1234/path",
      "http://127.0.0.1:1234?query=x", "http://127.0.0.1:1234#fragment",
    ] {
      let host = try descriptor(origin, token)
      #expect(!host.isValidPrivateHost)
      #expect(throws: (any Error).self) { try NativeConversationTransport(host: host) }
    }
    #expect(try descriptor("http://127.0.0.1:1234", token).isValidPrivateHost)
    #expect(
      try !descriptor("http://127.0.0.1:1234", token + "\r\nInjected: header").isValidPrivateHost)
  }
  @Test func endpointCannotEscapePrivateRoutes() {
    for id in ["", "../settings", "https://evil.example", "x%2f..", "a/b", "a?token=x", "a#b"] {
      #expect(NativeConversationTransport.Endpoint.cloud(id).path == nil)
    }
    #expect(NativeConversationTransport.Endpoint.runner(0).path == nil)
    #expect(
      NativeConversationTransport.Endpoint.cloud("provider-id").path
        == "api/cloud/provider-id/v1/responses")
  }
  private func descriptor(_ origin: String, _ token: String) throws -> StudioHost {
    try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": origin, "cookieName": "paddock_desktop_session", "session": token,
      ]))
  }
}
