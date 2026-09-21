import AppKit
import PaddockStudio
import SwiftUI
import Testing
import WebKit

@testable import PaddockUI

@Suite("Native response chrome", .serialized) @MainActor
struct ResponseChromeTests {
  @Test func upstreamOverloadIsReadableEvenWhenOldRelayTruncatedItsJSON() throws {
    let body: [String: Any] = [
      "error": [
        "message": "Provider returned error", "code": 429,
        "metadata": [
          "raw":
            "moonshotai/kimi-k3 is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations",
          "provider_name": "DeepInfra", "provider_error_code": "engine_overloaded",
          "limit_source": "upstream_provider_shared_pool",
          "user_id": "PRIVATE_ACCOUNT", "headers": ["Authorization": "PRIVATE_SECRET"],
        ],
      ]
    ]
    let full = String(
      decoding: try JSONSerialization.data(withJSONObject: body, options: .sortedKeys),
      as: UTF8.self)
    // Actual old wire ordering, which was clipped before JSON parsing.
    let old =
      #"provider error: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"moonshotai/kimi-k3 is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"DeepInfra","is_byok":false,"provider_error_code":"engine_overloaded","limit_source":"upstream_provider_shared_pool","remedy_hint":"#
    for raw in [full, old, String(old.prefix(416))] {
      let info = ResponseErrorPresentation(raw)
      #expect(info.title == "Provider temporarily busy")
      #expect(info.message.contains("DeepInfra") && info.message.contains("Retry shortly"))
      #expect(!info.message.contains("{") && !info.message.contains("moonshotai/"))
      #expect(info.actionURL?.absoluteString == "https://openrouter.ai/settings/integrations")
      #expect(!(info.message + (info.details ?? "")).contains("PRIVATE_"))
    }
    let broken = ResponseErrorPresentation(
      #"provider error: {"error":{"message":"Provider returned"#)
    #expect(!broken.message.contains("{"))
  }
  let ageError =
    #"provider error: {"error":{"message":"This model requires you to complete the following before use: 18+ age confirmation. Confirm at https://openrouter.ai/settings/preferences.","code":403,"metadata":{"missing_attestation_types":["age_18plus"],"routing_funnel":[{"step":"Initial Endpoints","endpoint_count":1}],"failed_routing_step":"Gate Endpoints with Attestations"}},"user_id":"PRIVATE_ACCOUNT","headers":{"Authorization":"PRIVATE_SECRET"}}"#

  @Test func ageGateOffersAnExplicitSafeActionWithoutDumpingAccountOrRoutingData() {
    let info = ResponseErrorPresentation(ageError)
    #expect(info.title == "Age confirmation required")
    #expect(info.message.contains("18 or older"))
    #expect(info.actionURL?.absoluteString == "https://openrouter.ai/settings/preferences")
    let displayed = info.message + (info.details ?? "")
    #expect(!displayed.contains("PRIVATE_"))
    #expect(!displayed.contains("routing_funnel"))
    #expect(info.details?.contains("Status: 403") == true)
  }

  @Test func ordinaryErrorsRemainReadableAndArbitraryLinksDoNotBecomeActions() {
    for (code, title) in [
      (401, "Provider sign-in required"), (402, "Provider credit required"),
      (403, "Model access denied"), (429, "Provider rate limit reached"),
      (503, "Provider temporarily unavailable"),
    ] {
      let info = ResponseErrorPresentation(
        "provider error: {\"error\":{\"message\":\"Read https://untrusted.invalid\",\"code\":\(code)}}"
      )
      #expect(info.title == title)
      #expect(info.message == "Read https://untrusted.invalid")
      #expect(info.actionURL == nil)
    }
    #expect(ResponseErrorPresentation("Connection interrupted").message == "Connection interrupted")
    #expect(ResponseErrorPresentation(#"{"error":"Unavailable"}"#).message == "Unavailable")
    #expect(
      ResponseErrorPresentation(#"{"user_id":"PRIVATE_ACCOUNT"}"#).message
        == "The provider could not complete this request.")
  }

  @Test func nativeErrorFitsNarrowCompareLanesInBothThemesWithoutAWebView() {
    _ = NSApplication.shared
    for dark in [false, true] {
      for width: CGFloat in [240, 600] {
        let host = NSHostingController(
          rootView: NativeResponseErrorView(error: ageError)
            .frame(width: width).environment(\.colorScheme, dark ? .dark : .light))
        let size = host.sizeThatFits(in: CGSize(width: width, height: 1000))
        #expect(size.width <= width + 1)
        #expect(size.height > 100 && size.height < 600)
        #expect(!descendants(host.view).contains { $0 is WKWebView })
      }
    }
  }

  @Test func toolTitlesMatchWebHumanizationButDoNotLoseAcronyms() {
    for (raw, server, expected) in [
      ("artifacts__artifact_create", "artifacts", "Artifact create"),
      ("mcp_search_tools", "", "Search tools"),
      ("github__getHTTPResponse", "github", "Get HTTPResponse"), ("read_file", "", "Read file"),
    ] {
      #expect(ToolCallPresentation.title(raw, server: server) == expected)
    }
  }

  private func descendants(_ view: NSView) -> [NSView] {
    [view] + view.subviews.flatMap(descendants)
  }
}
