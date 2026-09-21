import Foundation
import PaddockNativeMarkdown
import SwiftUI

struct ResponseErrorPresentation: Equatable {
  let title: String
  let message: String
  var actionTitle: String?
  var actionURL: URL?
  var details: String?

  init(_ raw: String) {
    let cleaned =
      raw.hasPrefix("provider error:")
      ? String(raw.dropFirst("provider error:".count)).trimmingCharacters(
        in: .whitespacesAndNewlines)
      : raw.trimmingCharacters(in: .whitespacesAndNewlines)
    let bounded = String(cleaned.prefix(128 * 1024))
    var object = (try? JSONSerialization.jsonObject(with: Data(bounded.utf8))) as? [String: Any]
    // Older tool-loop errors were sliced inside JSON at 400 characters. Recover
    // only complete, allowlisted fields; never display the broken JSON itself.
    if object == nil, bounded.hasPrefix("{") { object = Self.recover(bounded) }
    var error = object?["error"] as? [String: Any] ?? object
    for _ in 0..<2 {
      guard let nested = error?["message"] as? String,
        nested.trimmingCharacters(in: .whitespaces).hasPrefix("{"),
        let decoded = (try? JSONSerialization.jsonObject(with: Data(nested.utf8))) as? [String: Any]
      else { break }
      error = decoded["error"] as? [String: Any] ?? decoded
    }
    let code =
      (error?["code"] as? Int) ?? (error?["status"] as? Int)
      ?? (error?["code"] as? String).flatMap(Int.init)
    let metadata = error?["metadata"] as? [String: Any] ?? [:]
    let provider = (metadata["provider_name"] as? String).map { String($0.prefix(80)) }
    let upstreamCode = metadata["provider_error_code"] as? String
    let limit = metadata["limit_source"] as? String
    let rawDetail = metadata["raw"] as? String ?? ""
    let explanation =
      error?["message"] as? String ?? object?["error"] as? String
      ?? (object == nil && !bounded.hasPrefix("{")
        ? bounded : "The provider could not complete this request.")
    let missing = metadata["missing_attestation_types"] as? [String] ?? []
    let preferences = "https://openrouter.ai/settings/preferences"
    let ageConfirmation =
      code == 403
      && (missing.contains("age_18plus")
        || explanation.contains(preferences)
          && explanation.localizedCaseInsensitiveContains("age confirmation"))
    if ageConfirmation {
      title = "Age confirmation required"
      message =
        "OpenRouter requires you to confirm that you're 18 or older before using this model. Confirm in your account preferences, then retry this response."
      actionTitle = "Open OpenRouter preferences"
      actionURL = URL(string: preferences)
    } else if code == 429
      && (upstreamCode == "engine_overloaded" || limit == "upstream_provider_shared_pool"
        || rawDetail.localizedCaseInsensitiveContains("rate-limited upstream")
        || explanation.localizedCaseInsensitiveContains("rate-limited upstream"))
    {
      title = "Provider temporarily busy"
      message =
        "\(provider ?? "The upstream provider") is temporarily rate-limiting this model. Retry shortly, or choose another provider or model. This is a provider capacity limit, not a problem with your Mac."
      if metadata["action"] as? String == "openrouter_integrations"
        || rawDetail.contains("https://openrouter.ai/settings/integrations")
      {
        actionTitle = "OpenRouter provider keys"
        actionURL = URL(string: "https://openrouter.ai/settings/integrations")
      }
    } else {
      switch code {
      case 401: title = "Provider sign-in required"
      case 402: title = "Provider credit required"
      case 403: title = "Model access denied"
      case 429: title = "Provider rate limit reached"
      case let status? where (500...599).contains(status):
        title = "Provider temporarily unavailable"
      default: title = "Couldn't complete the response"
      }
      let readable = rawDetail.isEmpty ? explanation : rawDetail
      message =
        readable.trimmingCharacters(in: .whitespaces).hasPrefix("{")
        ? "The provider returned incomplete error information. Retry shortly or choose another model."
        : String(readable.prefix(600))
      if code == 402, explanation.contains("https://openrouter.ai/settings/credits") {
        actionTitle = "Open OpenRouter credits"
        actionURL = URL(string: "https://openrouter.ai/settings/credits")
      }
    }
    var lines: [String] = []
    if let code { lines.append("Status: \(code)") }
    if let provider { lines.append("Provider: \(provider)") }
    if let type = error?["type"] as? String { lines.append("Type: " + String(type.prefix(80))) }
    if let upstreamCode { lines.append("Provider code: " + String(upstreamCode.prefix(80))) }
    if limit == "upstream_provider_shared_pool" { lines.append("Limit: shared upstream capacity") }
    if let retry = error?["retry_after_seconds"] as? Int, retry > 0 {
      lines.append("Retry after: \(retry) seconds")
    }
    if !missing.isEmpty {
      lines.append("Required confirmation: " + missing.prefix(8).joined(separator: ", "))
    }
    details = lines.isEmpty ? nil : lines.joined(separator: "\n")
  }

  private static func recover(_ text: String) -> [String: Any] {
    func field(_ key: String) -> String? {
      let pattern = #""\#(key)"\s*:\s*("(?:[^"\\]|\\.)*")"#
      guard let regex = try? NSRegularExpression(pattern: pattern),
        let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
        let range = Range(match.range(at: 1), in: text)
      else { return nil }
      return
        (try? JSONSerialization.jsonObject(with: Data(text[range].utf8), options: .fragmentsAllowed))
        as? String
    }
    var error: [String: Any] = [:]
    var metadata: [String: Any] = [:]
    if let message = field("message") { error["message"] = message }
    if let regex = try? NSRegularExpression(pattern: #""(?:code|status)"\s*:\s*(\d{3})"#),
      let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)),
      let range = Range(match.range(at: 1), in: text), let code = Int(text[range])
    {
      error["code"] = code
    }
    for key in ["raw", "provider_name", "provider_error_code", "limit_source", "action"] {
      if let value = field(key) { metadata[key] = value }
    }
    error["metadata"] = metadata
    return error
  }
}

struct NativeResponseErrorView: View {
  let error: String
  var onRetry: (() -> Void)?
  var retryDisabled = false
  var body: some View {
    let info = ResponseErrorPresentation(error)
    VStack(alignment: .leading, spacing: 10) {
      Label(info.title, systemImage: "exclamationmark.circle")
        .font(.system(size: 12, weight: .semibold))
      NativeSelectableText(info.message, size: 12).fixedSize(horizontal: false, vertical: true)
      if let onRetry {
        Button("Retry response", systemImage: "arrow.clockwise", action: onRetry)
          .buttonStyle(FlatButtonStyle()).disabled(retryDisabled)
          .accessibilityIdentifier("provider-error-retry")
      }
      if let title = info.actionTitle, let url = info.actionURL {
        Link(destination: url) { Label(title, systemImage: "arrow.up.right") }
          .font(.system(size: 12)).buttonStyle(FlatButtonStyle())
      }
      if let details = info.details {
        DisclosureGroup("Technical details") {
          Text(verbatim: details).font(.system(size: 11, design: .monospaced))
            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
        }.font(.system(size: 11)).foregroundStyle(.secondary).tint(.secondary)
      }
    }.padding(14).frame(maxWidth: .infinity, alignment: .leading)
      .background(PaddockStyle.surface, in: RoundedRectangle(cornerRadius: 12))
  }
}
