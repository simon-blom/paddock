import Foundation

extension NativeStudioRuntime {
  private struct DocumentPage {
    let part: O
    let page: Int?
    var row: O {
      var value: O = [
        "attachmentId": part["attachmentId"] ?? .string(""),
        "name": part["name"] ?? .string("Document"),
        "state": .string("queued"), "text": .string(""),
      ]
      if let page { value["page"] = .number(Decimal(page)) }
      return value
    }
  }

  func runDocument(modelID: String, messageID: String) async throws -> Bool {
    guard let document,
      let plan = NativeDocumentPlan.make(document: document, capability: capability(modelID))
    else { return false }
    let cap = capability(modelID)
    let endpoint = try endpoint(model(modelID))
    let pageLimit = min(64, max(1, cap["pdf"]?["max_pages"]?.integer ?? 40))
    let maxPixels = min(
      32 * 1024 * 1024, NativeVisionBudget(value: cap["vision_budget"])?.maxPixels ?? 1_500_000)
    var pages: [DocumentPage] = []
    // At most one retained PDF at a time, and only the current page is rasterized.
    var retainedPDF: (id: String, source: NativeDocumentSource)?
    for part in plan.parts {
      try Task.checkCancellation()
      if NativeDocumentPlan.isPDF(part) {
        let count: Int
        if let known = part["pages"]?.integer {
          count = known
        } else {
          let id = try Self.id(part["attachmentId"])
          retainedPDF = nil
          let bytes = try await transport.bytes("api/attachments/\(id)", maximum: 100 * 1024 * 1024)
          let source = try await openPDF(bytes)
          count = source.pageCount
          retainedPDF = (id, source)
        }
        for page in try NativeDocumentPlan.pages(
          part["pageRange"]?.string, count: count, limit: pageLimit)
        {
          pages.append(DocumentPage(part: part, page: page))
        }
      } else {
        pages.append(DocumentPage(part: part, page: nil))
      }
      guard pages.count <= 64 else {
        throw ConversationFailure.invalid("Select up to 64 document pages per turn")
      }
    }
    guard !pages.isEmpty else { return false }
    reducers[messageID] = nil
    documentRuns[messageID] = NativeDocumentRunState(
      sourceID: plan.sourceID, pages: pages.map(\.row))
    defer {
      flushDocumentRuns()
      documentRuns[messageID] = nil
    }
    do {
      try updateMessage(messageID) { $0["run"] = .object(runSnapshot(modelID: modelID, body: [:])) }
      flushDocumentRuns()
      try await persist()
      await emit()
      for (index, page) in pages.enumerated() {
        try Task.checkCancellation()
        documentRuns[messageID]?.begin(index)
        schedulePublish()
        do {
          var image: O
          if let number = page.page {
            let id = try Self.id(page.part["attachmentId"])
            if retainedPDF?.id != id {
              retainedPDF = nil
              let bytes = try await transport.bytes(
                "api/attachments/\(id)", maximum: 100 * 1024 * 1024)
              retainedPDF = (id, try await openPDF(bytes))
            }
            guard let source = retainedPDF?.source, number <= source.pageCount else {
              throw ConversationFailure.invalid("The saved page is outside this PDF")
            }
            let raster = try await source.render(number, maxPixels)
            image = raster.image
            documentRuns[messageID]?.pages[index]["width"] = .number(Decimal(raster.width))
            documentRuns[messageID]?.pages[index]["height"] = .number(Decimal(raster.height))
          } else {
            retainedPDF = nil
            let mime = page.part["mime"]?.string ?? "image/jpeg"
            guard !mime.contains("tiff"),
              !(page.part["name"]?.string ?? "").lowercased().hasSuffix(".tif")
            else {
              throw ConversationFailure.invalid(
                "TIFF page decoding is not available. Attach a PDF or individual page images.")
            }
            var url = page.part["modelUrl"]?.string ?? page.part["dataUrl"]?.string
            if let id = page.part["attachmentId"]?.string, ConversationDocument.validID(id) {
              let bytes = try await transport.bytes(
                "api/attachments/\(id)", maximum: 100 * 1024 * 1024)
              url = "data:\(mime);base64,\(bytes.base64EncodedString())"
            }
            guard let url else {
              throw ConversationFailure.invalid("The original image is missing")
            }
            image = [
              "type": .string("input_image"), "image_url": .string(url),
              "detail": page.part["detail"] ?? .string("auto"),
            ]
          }
          try Task.checkCancellation()
          var content: [V] = [.object(image)]
          // Fixed-vocabulary decoders take the OCR object, not free-text prompts.
          if cap["ocr"]?.object == nil, !plan.instruction.isEmpty {
            content.append(
              .object(["type": .string("input_text"), "text": .string(plan.instruction)]))
          }
          let input: [V] = [
            .object([
              "role": .string("user"), "type": .string("message"), "content": .array(content),
            ])
          ]
          var body: O = [
            "model": .string(modelID), "stream": .bool(true), "input": .array(input),
            "include": .array([.string("message.output_text.logprobs")]),
          ]
          body["max_output_tokens"] = NativeReplyBudget.resolve(
            requested: maxTokens.integer, cloud: false, context: cap["max_ctx"]?.integer ?? 0
          ).map { .number(Decimal($0)) }
          var fields = document.fields
          if fields["ocrMode"]?.string == "multipage" { fields["ocrMode"] = nil }
          _ = NativeDocumentRequestOptions.apply(
            fields: fields, capability: cap, input: input, body: &body)
          try updateMessage(messageID) {
            $0["run"] = .object(runSnapshot(modelID: modelID, body: body))
          }
          let receive: @Sendable (O) async throws -> Void = { [weak self] event in
            try await self?.receiveDocument(event, messageID: messageID)
          }
          do {
            _ = try await transport.responses(endpoint: endpoint, body: body, receive: receive)
          } catch {
            // Only an explicit unsupported-include refusal permits one retry.
            // Never repeat a generated page or hide unrelated provider errors.
            guard Self.unsupportedDocumentLogprobs(error) else { throw error }
            body["include"] = nil
            _ = try await transport.responses(endpoint: endpoint, body: body, receive: receive)
          }
          let extracted =
            documentRuns[messageID]?.response.status == "completed"
            ? documentRuns[messageID]?.response.text ?? "" : ""
          let repetition = try await Task.detached(priority: .utility) {
            try NativeOCRQuality.repetitionRatio(extracted)
          }.value
          try Task.checkCancellation()
          try documentRuns[messageID]?.finish()
          if repetition > NativeOCRQuality.threshold {
            documentRuns[messageID]?.pages[index]["state"] = .string("review")
            documentRuns[messageID]?.pages[index]["note"] = .string(
              "Output is unusually repetitive. Review this page against the original.")
            documentRuns[messageID]?.pages[index]["repetitionRatio"] = .number(Decimal(repetition))
          }
        } catch {
          if Task.isCancelled || error is CancellationError { throw CancellationError() }
          documentRuns[messageID]?.fail(error.localizedDescription, index: index)
        }
        flushDocumentRuns()
        try await persist()
        await emit()
      }
      if let state = documentRuns[messageID] {
        let elapsed = starts[messageID].map { Self.seconds($0.duration(to: .now)) } ?? 0
        try updateMessage(messageID) {
          $0["usage"] = state.usage(seconds: elapsed).map(V.object)
          if state.pages.count == 1 { $0["ocr"] = state.pages.first?["ocr"] }
          if state.pages.contains(where: { $0["state"]?.string == "error" }) {
            $0["error"] = .string(
              "Some pages could not be read. Their errors are shown with the saved partial results."
            )
          }
        }
      }
    } catch {
      documentRuns[messageID]?.stopRemaining(
        Task.isCancelled || error is CancellationError
          ? "Stopped" : "Interrupted before this page completed")
      throw error
    }
    return true
  }

  func receiveDocument(_ event: O, messageID: String) async throws {
    try Task.checkCancellation()
    guard documentRuns[messageID] != nil else { throw ConversationFailure.stale }
    let elapsed = starts[messageID].map { Self.seconds($0.duration(to: .now)) } ?? 0
    try documentRuns[messageID]?.receive(event, elapsed: elapsed)
    // Enforce the aggregate quota while streaming, not only after completion.
    try documentRuns[messageID]?.flush()
    if ContinuousClock.now - lastCheckpoint >= .seconds(2) {
      flushDocumentRuns()
      lastCheckpoint = .now
      try await persist()
    }
    schedulePublish()
  }
  func flushDocumentRuns() {
    for (id, state) in documentRuns {
      try? updateMessage(id) {
        $0["docRun"] = state.snapshot
        $0["content"] = .array([.object(["type": .string("text"), "text": .string(state.text)])])
      }
    }
  }
  private static func unsupportedDocumentLogprobs(_ error: any Error) -> Bool {
    guard case ConversationFailure.invalid(let message) = error,
      let value = try? JSONDecoder().decode(V.self, from: Data(message.utf8)),
      value["code"]?.integer == 400,
      let detail = value["message"]?.string?.lowercased()
    else { return false }
    return (detail.contains("include") || detail.contains("logprobs"))
      && (detail.contains("unsupported") || detail.contains("unknown")
        || detail.contains("unrecognized") || detail.contains("not supported"))
  }
}
