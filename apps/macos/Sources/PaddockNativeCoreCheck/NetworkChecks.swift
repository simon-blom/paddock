import Foundation
import Network
import PaddockClient
import PaddockConversationCore

extension NativeCoreCheck {
  static func networkChecks() async throws {
    let fixture = try ResponseFixture()
    let port = try await fixture.start()
    defer { fixture.close() }
    let host = try JSONDecoder().decode(
      StudioHost.self,
      from: JSONEncoder().encode([
        "origin": "http://127.0.0.1:\(port)", "cookieName": "paddock_desktop_session",
        "session": String(repeating: "a", count: 64),
      ]))
    let transport = try NativeConversationTransport(host: host)
    let body: [String: ConversationValue] = [
      "model": .string("synthetic"), "stream": .bool(true), "input": .array([]),
    ]
    // Two simultaneous streams: proves independent reductions and no main
    // actor/network serialization. This is not a model throughput benchmark.
    async let first = transport.responses(endpoint: .runner(1), body: body) { _ in }
    async let second = transport.responses(endpoint: .cloud("fixture"), body: body) { _ in }
    let result = try await (first, second)
    try require(
      result.0.text == "Native tail 🦊" && result.1.text == result.0.text,
      "concurrent local/cloud transport")
    try require(result.0.reasoning == "Think", "streamed reasoning retained")
    // EOF without a terminal event cannot be reported as a completed reply.
    do {
      _ = try await transport.responses(endpoint: .runner(2), body: body) { _ in }
      throw ConversationFailure.invalid("Truncated response was accepted")
    } catch ConversationFailure.interrupted {}
    // Redirects must not follow even to another path on the private host.
    do {
      _ = try await transport.responses(endpoint: .runner(3), body: body) { _ in }
      throw ConversationFailure.invalid("Redirect was followed")
    } catch ConversationFailure.http(302) {}
    // A task cancelled by its consumer closes its underlying URLSession task.
    let waiting = Task {
      try await transport.responses(endpoint: .runner(4), body: body) { _ in
        try await Task.sleep(for: .seconds(30))
      }
    }
    try await Task.sleep(for: .milliseconds(80))
    waiting.cancel()
    do {
      _ = try await waiting.value
      throw ConversationFailure.invalid("Cancelled stream completed")
    } catch is CancellationError {} catch let error as URLError where error.code == .cancelled {}
    await transport.close()
    print(
      "PASS: native local/cloud-shaped streams, final-only tail, interruption, redirect refusal and cancellation"
    )
  }
}

/// Explicit synthetic inference on ephemeral loopback only; never registers a
/// model, uses a provider, touches real conversations, or requests a key.
private final class ResponseFixture: @unchecked Sendable {
  private let listener: NWListener
  private let queue = DispatchQueue(label: "paddock.native-core.fixture")
  // All mutable state is queue-confined, including teardown.
  private var connections = [NWConnection]()
  private var startFinished = false
  init() throws {
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = .hostPort(host: .ipv4(.loopback), port: .any)
    listener = try NWListener(using: parameters)
  }
  func start() async throws -> UInt16 {
    try await withCheckedThrowingContinuation { continuation in
      queue.async { [self] in
        listener.stateUpdateHandler = { [self] state in
          guard !startFinished else { return }
          switch state {
          case .ready:
            startFinished = true
            if let port = listener.port {
              continuation.resume(returning: port.rawValue)
            } else {
              continuation.resume(throwing: ConversationFailure.invalid("Missing fixture port"))
            }
          case .failed(let error):
            startFinished = true
            continuation.resume(throwing: error)
          default: break
          }
        }
        listener.newConnectionHandler = { [self] connection in
          connections.append(connection)
          connection.start(queue: queue)
          read(connection, previous: Data())
        }
        listener.start(queue: queue)
      }
    }
  }
  func close() {
    queue.async { [self] in
      listener.cancel()
      for connection in connections { connection.cancel() }
      connections.removeAll()
    }
  }
  private func read(_ connection: NWConnection, previous: Data) {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) {
      [self] bytes, _, done, failure in
      var received = previous
      received.append(bytes ?? Data())
      guard failure == nil, received.count <= 128 * 1024 else {
        connection.cancel()
        return
      }
      guard let end = received.range(of: Data("\r\n\r\n".utf8)) else {
        if done { connection.cancel() } else { read(connection, previous: received) }
        return
      }
      let headers = String(decoding: received[..<end.lowerBound], as: UTF8.self).lowercased()
      let length =
        headers.components(separatedBy: "\r\n").first { $0.hasPrefix("content-length:") }
        .flatMap { Int($0.dropFirst("content-length:".count).trimmingCharacters(in: .whitespaces)) }
        ?? 0
      if received.count < end.upperBound + length {
        read(connection, previous: received)
        return
      }
      let cookie = "cookie: paddock_desktop_session=" + String(repeating: "a", count: 64)
      guard headers.contains(cookie), headers.hasPrefix("post /api/") else {
        respond(connection, status: "401 Unauthorized", type: "text/plain", body: "")
        return
      }
      if headers.contains("/runners/3/") {
        respond(
          connection, status: "302 Found", type: "text/plain", body: "",
          extra: "Location: /must-not-follow\r\n")
        return
      }
      var stream =
        "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"Think\"}\r\n\r\n"
        + "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Native\"}\r\n\r\n"
      if !headers.contains("/runners/2/") {
        stream +=
          "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Native tail 🦊\"}]}]}}\r\n\r\n"
      }
      respond(connection, status: "200 OK", type: "text/event-stream", body: stream)
    }
  }
  private func respond(
    _ connection: NWConnection, status: String, type: String, body: String, extra: String = ""
  ) {
    let response =
      "HTTP/1.1 \(status)\r\nContent-Type: \(type)\r\n\(extra)Content-Length: \(body.utf8.count)\r\nConnection: close\r\n\r\n\(body)"
    connection.send(
      content: Data(response.utf8), completion: .contentProcessed { _ in connection.cancel() })
  }
}
