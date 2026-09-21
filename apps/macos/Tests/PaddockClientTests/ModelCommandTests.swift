import Foundation
import Testing

@testable import PaddockClient

@Suite("Typed native model commands")
struct ModelCommandTests {
  @Test func editIsAnAllowlistedPatchAndPollingCarriesOnlyReceiptIdentity() throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let edit = try #require(
      JSONSerialization.jsonObject(
        with: encoder.encode(
          ModelCommand.edit(
            port: 12345, revision: "reviewed", pid: 42,
            changes: [.maxCtx(nil), .host("0.0.0.0")], apply: .restart, allowNetwork: true)
        )) as? [String: Any])
    #expect(edit["kind"] as? String == "edit" && edit["apply"] as? String == "restart")
    #expect(edit["allow_network"] as? Bool == true && edit["pid"] as? Int == 42)
    #expect(edit["revision"] as? String == "reviewed")
    let changes = try #require(edit["changes"] as? [[String: Any]])
    #expect(changes.count == 2 && changes[0]["value"] is NSNull)
    for field in ["content", "path", "api_key", "url", "evict"] { #expect(edit[field] == nil) }
    let poll = try #require(
      JSONSerialization.jsonObject(
        with: encoder.encode(
          ModelCommand.poll(id: 7))) as? [String: Any])
    #expect(Set(poll.keys) == ["kind", "id"] && poll["id"] as? Int == 7)
    let start = try #require(
      JSONSerialization.jsonObject(
        with: encoder.encode(
          ModelCommand.start(port: 12345, revision: "reviewed"))) as? [String: Any])
    #expect(start["allow_network"] as? Bool == false)
  }

  @Test func createUsesRustFieldNamesAndOmitsPrivilegedControls() throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(
      ModelCommand.create(
        CreateEndpointRequest(
          model: "qwen3.8-27b", artifact: "mlx-4bit", port: 12345, maxCtx: 4096, maxBatch: 4)))
    let value = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    #expect(value["kind"] as? String == "create")
    #expect(value["max_ctx"] as? Int == 4096)
    #expect(value["max_batch"] as? Int == 4)
    #expect(value["port"] as? Int == 12345)
    for field in ["host", "api_key", "pull", "evict", "url", "runner_bin", "persist"] {
      #expect(value[field] == nil)
    }
  }

  @Test func startAndStopCarryStaleStateGuards() throws {
    let start =
      try JSONSerialization.jsonObject(
        with: JSONEncoder().encode(ModelCommand.start(port: 12345, revision: "hash")))
      as? [String: Any]
    let stop =
      try JSONSerialization.jsonObject(
        with: JSONEncoder().encode(ModelCommand.stop(port: 12345, pid: 42))) as? [String: Any]
    #expect(start?["revision"] as? String == "hash")
    #expect(stop?["pid"] as? Int == 42)
  }
}
