import Foundation

extension NativeStudioRuntime {
  static func visibleToolCall(_ call: V) -> Bool {
    guard let type = call["type"]?.string else { return call["name"]?.string?.isEmpty == false }
    return type == "mcp_call" || type == "mcp_approval_request"
  }
  static func toolArtifactID(_ call: V) -> String? {
    let name = call["name"]?.string ?? ""
    guard name.hasPrefix("artifacts__artifact_") || name.hasPrefix("artifact_") else { return nil }
    let text = "\(call["arguments"]?.string ?? "") \(call["output"]?.string ?? "")"
    guard let range = text.range(of: "art_[0-9a-f]{12}", options: .regularExpression) else {
      return nil
    }
    return String(text[range])
  }
  func refreshTools() async throws {
    connectors = try await transport.api("api/connectors").array?.compactMap(\.object) ?? []
    let id = selected.first ?? ""
    let model = models.first { $0["id"]?.string == id }
    var sources: [(String, String, O)] = [("artifacts", "", ["builtin": .string("artifacts")])]
    for label in caps[id]?["mcp_servers"]?.array?.compactMap(\.string) ?? [] {
      if let port = model?["port"] {
        sources.append((label, "", ["port": port, "label": .string(label)]))
      }
    }
    for c in connectors where c["system"]?.bool != true {
      guard let label = c["label"]?.string, let cid = c["id"]?.string,
        !sources.contains(where: { $0.0 == label })
      else { continue }
      sources.append((label, cid, ["connector_id": .string(cid)]))
    }
    var groups: [O] = []
    for (label, cid, query) in sources {
      let result = try? await transport.api("api/mcp/tools", method: "POST", body: .object(query))
      groups.append([
        "id": .string(label), "label": .string(label == "artifacts" ? "Artifacts" : label),
        "connectorId": .string(cid),
        "tools": result?["tools"] ?? .array([]),
        "status": .string(result?["ok"]?.bool == true ? "ok" : "error"),
        "instructions": result?["instructions"] ?? .string(""),
      ])
    }
    toolGroups = groups
  }
  func toolChecked(_ label: String, connector: String, tool: String? = nil) -> Bool {
    let selection = document?.fields["toolSelection"]?.object ?? ["mode": .string("all")]
    if selection["mode"]?.string == "all" {
      return !connector.isEmpty
        && (document?.fields["connectorIds"]?.array ?? []).contains(.string(connector))
    }
    return (selection["picks"]?.array ?? []).contains {
      $0["label"]?.string == label && ($0["tool"] == nil || $0["tool"]?.string == tool)
    }
  }
  func projectTools() -> [V] {
    let terms = toolQuery.lowercased().split(whereSeparator: \.isWhitespace).map(String.init)
    return toolGroups.compactMap { group in
      var row = group
      let label = group["id"]?.string ?? ""
      let cid = group["connectorId"]?.string ?? ""
      let tools = group["tools"]?.array ?? []
      let filtered = tools.filter { item in
        terms.allSatisfy {
          "\(label) \(item["name"]?.string ?? "") \(item["description"]?.string ?? "")"
            .localizedCaseInsensitiveContains($0)
        }
      }
      if !terms.isEmpty && filtered.isEmpty { return nil }
      row["tools"] = .array(
        filtered.map { item in
          var value = item.object ?? [:]
          value["selected"] = .bool(toolChecked(label, connector: cid, tool: value["name"]?.string))
          return .object(value)
        })
      let count = tools.filter { toolChecked(label, connector: cid, tool: $0["name"]?.string) }
        .count
      row["checked"] = .string(
        toolChecked(label, connector: cid) ? "all" : count > 0 ? "some" : "none")
      row["total"] = .number(Decimal(tools.count))
      row["selectedCount"] = .number(Decimal(count))
      return .object(row)
    }
  }
  func changeTools(_ p: O) async throws {
    let action = p["action"]?.string ?? ""
    let before = document
    var selection = document?.fields["toolSelection"]?.object ?? ["mode": .string("all")]
    var ids = document?.fields["connectorIds"]?.array ?? []
    if action == "all" {
      selection = ["mode": .string("all")]
    } else if action == "clear" {
      selection = ["mode": .string("custom"), "picks": .array([])]
    } else {
      let label = try Self.text(p["label"], limit: 128)
      guard let group = toolGroups.first(where: { $0["id"]?.string == label }) else {
        throw ConversationFailure.stale
      }
      let cid = group["connectorId"]?.string ?? ""
      if action == "group", selection["mode"]?.string == "all", !cid.isEmpty {
        if ids.contains(.string(cid)) {
          ids.removeAll { $0 == .string(cid) }
        } else {
          ids.append(.string(cid))
        }
      } else {
        var picks =
          selection["picks"]?.array
          ?? connectors.filter { ids.contains($0["id"] ?? .null) }.map {
            V.object(["label": $0["label"]!])
          }
        if action == "group" {
          let all = picks.contains { $0["label"]?.string == label && $0["tool"] == nil }
          picks.removeAll { $0["label"]?.string == label }
          if !all { picks.append(.object(["label": .string(label)])) }
        } else if action == "tool" {
          let name = try Self.text(p["tool"], limit: 256)
          let names = group["tools"]?.array?.compactMap { $0["name"]?.string } ?? []
          guard names.contains(name) else { throw ConversationFailure.stale }
          if picks.contains(where: { $0["label"]?.string == label && $0["tool"] == nil }) {
            picks.removeAll { $0["label"]?.string == label }
            picks += names.filter { $0 != name }.map {
              .object(["label": .string(label), "tool": .string($0)])
            }
          } else if picks.contains(where: {
            $0["label"]?.string == label && $0["tool"]?.string == name
          }) {
            picks.removeAll { $0["label"]?.string == label && $0["tool"]?.string == name }
          } else {
            picks.append(.object(["label": .string(label), "tool": .string(name)]))
          }
        } else {
          throw ConversationFailure.invalid("Invalid tool action")
        }
        selection = ["mode": .string("custom"), "picks": .array(picks)]
      }
    }
    try change {
      $0["toolSelection"] = .object(selection)
      $0["connectorIds"] = .array(ids)
    }
    do { try await persist() } catch {
      document = before
      throw error
    }
  }
  func requestTools(modelID: String) async -> [V] {
    let fields = document?.fields ?? [:]
    let selection = fields["toolSelection"]?.object ?? ["mode": .string("all")]
    let all = selection["mode"]?.string == "all"
    let picks = selection["picks"]?.array ?? []
    let origin = await transport.origin.absoluteString.trimmingCharacters(
      in: CharacterSet(charactersIn: "/"))
    let cap = capability(modelID)
    let port = models.first { $0["id"]?.string == modelID }?["port"]
    var sources: [O] = [
      [
        "server_label": .string("artifacts"), "server_url": .string("\(origin)/api/mcp/artifacts"),
        "require_approval": .string("never"),
        "headers": .object([
          "x-paddock-conversation": fields["id"]!, "x-paddock-model": .string(modelID),
        ]),
      ]
    ]
    if !graphGrounding.isEmpty {
      sources.append([
        "server_label": .string("graph"), "server_url": .string("\(origin)/api/mcp/graph"),
        "require_approval": .string("never"),
        "headers": .object([
          "x-paddock-conversation": fields["id"]!, "x-paddock-model": .string(modelID),
        ]),
      ])
    }
    for label in cap["mcp_servers"]?.array ?? [] { sources.append(["server_label": label]) }
    let armed = fields["connectorIds"]?.array ?? []
    for c in connectors where c["system"]?.bool != true && armed.contains(c["id"] ?? .null) {
      guard let label = c["label"], !sources.contains(where: { $0["server_label"] == label }) else {
        continue
      }
      sources.append([
        "server_label": label, "server_url": c["url"] ?? .string(""),
        "require_approval": .string(
          (c["ports"]?.array ?? []).contains(port ?? .null) ? "never" : "always"),
      ])
    }
    var tools: [V] = []
    for var source in sources {
      let chosen = picks.filter { $0["label"] == source["server_label"] }
      if !all && chosen.isEmpty { continue }
      if !all && !chosen.contains(where: { $0["tool"] == nil }) {
        source["allowed_tools"] = .array(chosen.compactMap { $0["tool"] })
      }
      source["type"] = .string("mcp")
      tools.append(.object(source))
    }
    if cap["web_search"]?.bool == true, fields["webSearchEnabled"]?.bool != false {
      tools.append(.object(["type": .string("web_search")]))
    }
    if cap["current_time"]?.bool == true || modelID.hasPrefix("cloud:") {
      tools.append(
        .object(["type": .string("current_time"), "timezone": .string(TimeZone.current.identifier)])
      )
    }
    return tools
  }
  func applyOutputItem(_ item: O, messageID: String, done: Bool) throws {
    let type = item["type"]?.string ?? ""
    if type == "compaction", done, let id = item["id"], let content = item["encrypted_content"],
      selected.count == 1,
      let anchor = document?.activeMessages.last(where: { $0["role"]?.string == "user" })?["id"]
    {
      try change {
        $0["serverCompaction"] = .object([
          "id": id, "content": content, "tailStartId": anchor, "at": Self.now,
          "model": $0["model"] ?? .null,
        ])
      }
      return
    }
    // Match Web Studio's applyMcpItem: discovery/listing is protocol setup,
    // not a model invocation. A real mcp_search_tools call still appears.
    guard ["mcp_call", "mcp_approval_request", "web_search_call"].contains(type) else { return }
    try updateMessage(messageID) { message in
      let key = type == "web_search_call" ? "webSearches" : "toolCalls"
      var calls = message[key]?.array ?? []
      let id =
        (type == "mcp_approval_request" ? item["call_id"] : nil) ?? item["id"]
        ?? .string(UUID().uuidString)
      let index = calls.firstIndex { $0["id"] == id }
      var call = index.flatMap { calls[$0].object } ?? [:]
      call.merge(item) { _, new in new }
      call["id"] = id
      call["serverLabel"] = item["server_label"] ?? call["serverLabel"]
      if type == "mcp_approval_request" {
        call["approvalId"] = done ? nil : item["id"]
        call["status"] = .string(
          done ? item["status"]?.string == "denied" ? "denied" : "in_progress" : "pending")
      } else {
        call["status"] = item["status"] ?? .string(done ? "completed" : "in_progress")
        if type == "mcp_call" { call["approvalId"] = nil }
      }
      if type == "web_search_call" {
        call["query"] = item["action"]?["query"] ?? .string("")
        call["sources"] = item["action"]?["sources"] ?? .array([])
        call["provider"] = item["paddock_provider"]
      }
      if let value = call["error"], value != .null, value.string == nil {
        call["error"] = .string("Tool failed")
      } else if call["error"] == .null {
        call["error"] = nil
      }
      if let index { calls[index] = .object(call) } else { calls.append(.object(call)) }
      message[key] = .array(calls)
    }
  }
  func approveTool(_ p: O) async throws {
    guard p["conversationId"]?.string == document?.id, p["leafId"] == document?.fields["leafId"],
      let message = document?.activeMessages.first(where: { $0["id"] == p["messageId"] }),
      let approval = p["approvalId"]?.string, let approve = p["approve"]?.bool,
      (message["toolCalls"]?.array ?? []).contains(where: {
        $0["approvalId"]?.string == approval && $0["id"] == p["callId"]
          && $0["status"]?.string == "pending"
      })
    else { throw ConversationFailure.stale }
    _ = try Self.id(.string(approval))
    let model = try model(message["model"]!.string!)
    let path =
      model["port"]?.integer.map { "api/runners/\($0)/mcp-approvals/\(approval)" }
      ?? "api/cloud/mcp-approvals/\(approval)"
    _ = try await transport.api(path, method: "POST", body: .object(["approve": .bool(approve)]))
  }
}
