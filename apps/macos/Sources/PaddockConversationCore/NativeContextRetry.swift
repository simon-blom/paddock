import Foundation

extension NativeContextPlan {
  /// Only explicit, numeric context-overflow refusals permit one corrected
  /// request. Not rate limits, auth errors, timeouts or an interrupted stream.
  static func correctedLimit(error: Error, previous: Int, ceiling: Int?) -> Int? {
    guard let raw = try? JSONDecoder().decode(O.self, from: Data(error.localizedDescription.utf8)),
      [400, 413, 422].contains(raw["code"]?.integer ?? 0), let message = raw["message"]?.string
    else { return nil }
    let patterns = [
      (#"exceed context limit:\s*(\d+)\s*\+\s*(\d+)\s*>\s*(\d+)"#, 3, 1),
      (
        #"maximum context length is (\d+) tokens.*?requested (\d+) tokens \((\d+) in the messages"#,
        1, 3
      ),
    ]
    for (pattern, limitIndex, inputIndex) in patterns {
      guard
        let regex = try? NSRegularExpression(
          pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]),
        let match = regex.firstMatch(
          in: message, range: NSRange(message.startIndex..., in: message)),
        let lr = Range(match.range(at: limitIndex), in: message),
        let ir = Range(match.range(at: inputIndex), in: message),
        let limit = Int(message[lr]), let input = Int(message[ir]),
        (1...1_000_000_000).contains(limit), (0...1_000_000_000).contains(input)
      else { continue }
      let fits = min(limit - input, ceiling.flatMap { $0 > 0 ? $0 : nil } ?? Int.max)
      return fits >= NativeReplyBudget.minimumUsefulReply && fits < previous ? fits : nil
    }
    return nil
  }
}

extension NativeStudioRuntime {
  func responseWithContextRetry(
    endpoint: NativeConversationTransport.Endpoint, body: O,
    modelID: String, messageID: String, continuing: Bool
  ) async throws -> ResponseAccumulator {
    let receive: @Sendable (O) async throws -> Void = { [weak self] event in
      try await self?.receive(event, messageID: messageID)
    }
    do {
      return try await transport.responses(endpoint: endpoint, body: body, receive: receive)
    } catch {
      guard !Task.isCancelled, !requestEvents.contains(messageID),
        models.first(where: { $0["id"]?.string == modelID })?["endpoint"] != nil,
        let fits = NativeContextPlan.correctedLimit(
          error: error, previous: body["max_output_tokens"]?.integer ?? Int.max,
          ceiling: capability(modelID)["default_max_output_tokens"]?.integer)
      else { throw error }
      var retry = body
      retry["max_output_tokens"] = .number(Decimal(fits))
      if !continuing {
        let run = runSnapshot(modelID: modelID, body: retry)
        try updateMessage(messageID) { $0["run"] = .object(run) }
      }
      return try await transport.responses(endpoint: endpoint, body: retry, receive: receive)
    }
  }
}
