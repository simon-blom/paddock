import Foundation

/// Task-owned multipart spool: at most one downloaded original is held while
/// building the form, not an array plus a second concatenated copy. The final
/// request maps the private file; cleanup runs on success, failure and cancel.
final class NativeImageMultipart {
  static let maximumBytes = 192 * 1024 * 1024  // Rust relay/runner body limit.
  static let maximumImageBytes = 100 * 1024 * 1024  // Native attachment limit.
  let boundary = UUID().uuidString
  let directory: URL
  let url: URL
  private var file: FileHandle?
  private(set) var count = 0
  private var images = 0
  private let limit: Int

  init(fields: NativeImageGeneration.O, maximumBytes: Int = NativeImageMultipart.maximumBytes)
    throws
  {
    guard (1...Self.maximumBytes).contains(maximumBytes) else { throw ConversationFailure.tooLarge }
    limit = maximumBytes
    directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "paddock-image-edit-\(UUID().uuidString)", isDirectory: true)
    url = directory.appendingPathComponent("request.multipart")
    try FileManager.default.createDirectory(
      at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
    do {
      guard
        FileManager.default.createFile(
          atPath: url.path, contents: nil, attributes: [.posixPermissions: 0o600])
      else { throw ConversationFailure.invalid("Could not prepare the image upload") }
      file = try FileHandle(forWritingTo: url)
      guard try JSONEncoder().encode(fields).count <= 256 * 1024 else {
        throw ConversationFailure.tooLarge
      }
      let allowed: Set<String> = [
        "model", "prompt", "n", "seed", "output_format", "size", "quality", "background",
        "steps", "stream", "partial_images",
      ]
      for (key, value) in fields.sorted(by: { $0.key < $1.key }) {
        guard allowed.contains(key) else {
          throw ConversationFailure.invalid("Invalid image field")
        }
        let text: String
        switch value {
        case .string(let value): text = value
        case .bool(let value): text = value ? "true" : "false"
        case .number(let value): text = NSDecimalNumber(decimal: value).stringValue
        default: throw ConversationFailure.invalid("Invalid image field value")
        }
        try append(
          Data(
            "--\(boundary)\r\nContent-Disposition: form-data; name=\"\(key)\"\r\n\r\n\(text)\r\n"
              .utf8))
      }
    } catch {
      discard()
      throw error
    }
  }
  var remainingImageBytes: Int {
    min(Self.maximumImageBytes, max(0, limit - count - 1024))
  }
  func add(_ bytes: Data, mime: String) throws {
    guard !bytes.isEmpty, bytes.count <= remainingImageBytes, images < 10 else {
      throw ConversationFailure.tooLarge
    }
    guard Self.validMIME(mime) else {
      throw ConversationFailure.invalid("Invalid reference image type")
    }
    images += 1
    try append(
      Data(
        "--\(boundary)\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"reference-\(images)\"\r\nContent-Type: \(mime)\r\n\r\n"
          .utf8))
    try append(bytes)
    try append(Data("\r\n".utf8))
  }
  func finish() throws -> Data {
    guard images > 0, file != nil else {
      throw ConversationFailure.invalid("Attach a reference picture")
    }
    try append(Data("--\(boundary)--\r\n".utf8))
    try file?.close()
    file = nil
    try Task.checkCancellation()
    return try Data(contentsOf: url, options: .alwaysMapped)
  }
  private func append(_ bytes: Data) throws {
    guard let file, bytes.count <= limit - count else {
      throw ConversationFailure.tooLarge
    }
    for offset in stride(from: 0, to: bytes.count, by: 64 * 1024) {
      try Task.checkCancellation()
      let start = bytes.index(bytes.startIndex, offsetBy: offset)
      let end = bytes.index(bytes.startIndex, offsetBy: min(bytes.count, offset + 64 * 1024))
      try file.write(contentsOf: bytes[start..<end])
    }
    count += bytes.count
  }
  static func validMIME(_ mime: String) -> Bool {
    mime.hasPrefix("image/") && (7...127).contains(mime.utf8.count)
      && mime.dropFirst(6).utf8.allSatisfy {
        (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0)
          || [43, 45, 46].contains($0)
      }
  }
  func discard() {
    try? file?.close()
    file = nil
    try? FileManager.default.removeItem(at: directory)
  }
  deinit { discard() }
}

extension NativeConversationTransport {
  func imageEditBody(fields: NativeImageGeneration.O, references: [NativeImageGeneration.O])
    async throws
    -> NativeImageMultipart
  {
    guard (1...10).contains(references.count) else { throw ConversationFailure.tooLarge }
    let form = try NativeImageMultipart(fields: fields)
    for part in references {
      try Task.checkCancellation()
      let data: Data
      let mime: String
      let id = part["attachmentId"]?.string ?? ""
      if !id.isEmpty {
        guard ConversationDocument.validID(id) else {
          throw ConversationFailure.invalid("Invalid reference attachment")
        }
        mime = part["mime"]?.string ?? "image/png"
        do {
          data = try await bytes("api/attachments/\(id)", maximum: form.remainingImageBytes)
        } catch ConversationFailure.http(404) {
          throw ConversationFailure.invalid(
            "The reference picture is no longer available. Reattach its original file")
        }
      } else {
        // Originals only; never send a thumbnail or fetch an external URL with
        // the local session cookie. Restored web inline originals are supported.
        guard let source = part["dataUrl"]?.string ?? part["modelUrl"]?.string,
          source.utf8.count <= (form.remainingImageBytes + 2) / 3 * 4 + 256,
          source.hasPrefix("data:image/"), let comma = source.firstIndex(of: ",")
        else {
          throw ConversationFailure.invalid(
            "The reference picture is unavailable. Reattach its original file")
        }
        let header = String(source[..<comma].dropFirst(5))
        guard header.hasSuffix(";base64") else {
          throw ConversationFailure.invalid("Invalid reference image data")
        }
        mime = String(header.dropLast(7))
        guard let decoded = Data(base64Encoded: String(source[source.index(after: comma)...]))
        else {
          throw ConversationFailure.invalid("Invalid reference image data")
        }
        data = decoded
      }
      try form.add(data, mime: mime)
    }
    return form
  }
}
