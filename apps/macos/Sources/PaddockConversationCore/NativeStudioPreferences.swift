import Foundation

extension NativeStudioRuntime {
  func preferencePresentation() -> O {
    return [
      "maxTokens": maxTokens,
      "maxToolCalls": preferences["pk_max_tool_calls"]?.string.flatMap(Int.init).map {
        .number(Decimal($0))
      } ?? .null,
      "summarize": .bool(preferenceBool("summarize", fallback: true)),
      "autoTitle": .bool(preferenceBool("auto_title", fallback: true)),
      "markUnsure": .bool(preferenceBool("mark_unsure", fallback: true)),
      "mapTiles": preferences["pk_map_tiles"] ?? .string(""), "mapHost": .string(""),
      "layout": .object([
        "sections": .array(
          [
            ("maxTokens", "Reply limit"), ("maxToolCalls", "Tools per reply"),
            ("summarize", "Summarize older messages"), ("microphone", "Microphone"),
            ("mapTiles", "Map tiles"),
          ].map { .object(["id": .string($0.0), "title": .string($0.1)]) }),
        "replyLimit": .object(["maximum": .number(1_048_576)]),
        "toolStops": .array(
          [0, 5, 10, 25, 50, 100].map {
            .object([
              "value": .number(Decimal($0)),
              "label": .string($0 == 0 ? "Server default" : "\($0) tool calls"),
            ])
          }),
      ]),
    ]
  }
  func savePreferences(_ p: O) async throws -> O {
    guard let changes = p["changes"]?.object, let expected = p["expected"]?.object else {
      throw ConversationFailure.invalid("Invalid preference update")
    }
    let before = preferencePresentation()
    var next = preferences
    let keys = [
      "maxTokens": "max_tokens", "maxToolCalls": "max_tool_calls", "summarize": "summarize",
      "autoTitle": "auto_title", "markUnsure": "mark_unsure", "mapTiles": "map_tiles",
    ]
    for (key, value) in changes {
      guard let name = keys[key], expected[key] == before[key] else {
        throw ConversationFailure.stale
      }
      if key == "maxTokens" || key == "maxToolCalls" {
        guard
          value == .null
            || value.integer.map({ (1...(key == "maxTokens" ? 1_048_576 : 10000)).contains($0) })
              == true
        else { throw ConversationFailure.invalid("Invalid limit") }
        next["pk_\(name)"] = .string(value.integer.map(String.init) ?? "null")
      } else if key == "mapTiles" {
        let address = try Self.text(value, limit: 4096)
        if !address.isEmpty {
          let test = address.replacingOccurrences(
            of: "\\{[^}]*\\}", with: "0", options: .regularExpression)
          guard let url = URLComponents(string: test), ["http", "https"].contains(url.scheme),
            url.host != nil, url.user == nil, url.password == nil
          else {
            throw ConversationFailure.invalid("Use an absolute HTTP(S) map URL without credentials")
          }
        }
        next["pk_\(name)"] = .string(address)
      } else {
        guard let flag = value.bool else {
          throw ConversationFailure.invalid("Invalid preference switch")
        }
        next["pk_\(name)"] = .string(flag ? "on" : "off")
      }
    }
    _ = try await transport.api(
      "api/settings", method: "PUT", body: .object(["macos_studio_preferences": .object(next)]))
    preferences = next
    return ["preferences": .object(preferencePresentation())]
  }
  func promptCommand(_ kind: String, _ p: O) async throws -> O {
    if kind == "promptSave" {
      let id = try Self.id(p["id"])
      let name = try Self.text(p["name"], limit: 512)
      let body = try Self.text(p["body"], limit: 128 * 1024)
      guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      else { throw ConversationFailure.invalid("Enter a name and prompt text") }
      return try await transport.api(
        "api/prompts", method: "POST",
        body: .object([
          "id": .string(id), "name": .string(name), "body": .string(body),
          "revision": p["revision"] ?? .string(""),
        ])
      ).object ?? [:]
    }
    if kind == "promptDelete" {
      let id = try Self.id(p["id"])
      _ = try await transport.api(
        "api/prompts/\(id)", method: "DELETE",
        query: ["revision": try Self.text(p["revision"], limit: 64)])
      return ["deleted": .string(id)]
    }
    let rows = try await transport.api("api/prompts").array ?? []
    if kind == "promptGet" {
      let id = try Self.id(p["id"])
      guard let row = rows.first(where: { $0["id"]?.string == id }) else {
        throw ConversationFailure.stale
      }
      return ["prompt": row]
    }
    let query = try Self.text(p["search"] ?? .string(""), limit: 512)
    let matches = rows.filter {
      query.isEmpty
        || "\($0["name"]?.string ?? "")\n\($0["body"]?.string ?? "")"
          .localizedCaseInsensitiveContains(query)
    }
    let page = min(max(0, p["page"]?.integer ?? 0), max(0, (matches.count - 1) / 40))
    return [
      "library": .object([
        "page": .number(Decimal(page)), "pageSize": .number(40),
        "total": .number(Decimal(rows.count)), "matched": .number(Decimal(matches.count)),
        "rows": .array(
          matches.dropFirst(page * 40).prefix(40).map {
            .object([
              "id": $0["id"]!, "name": $0["name"]!,
              "preview": .string(String(($0["body"]?.string ?? "").prefix(256))),
              "revision": $0["revision"] ?? .string(""), "updatedAt": $0["updatedAt"] ?? .number(0),
            ])
          }),
      ])
    ]
  }
}
