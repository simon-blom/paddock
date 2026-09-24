import CryptoKit
import Foundation

/// The same persisted image recipe and seed policy as web Studio, independent
/// of text sampling. Endpoint defaults stay absent from the request.
enum NativeImageGeneration {
  typealias V = ConversationValue
  typealias O = [String: V]
  static let defaults: O = [
    "size": .string("auto"), "quality": .string("auto"), "steps": .null,
    "seed": .string("thread"), "n": .number(1), "format": .string("png"),
    "background": .string("auto"), "previews": .number(2),
  ]
  /// Text-to-image accumulates descriptions, as in web Studio's promptFor.
  /// Restored/retried turns must not bypass the composer's attachment guard:
  /// text-only endpoints must never silently discard an attached reference.
  static func textPrompt(_ messages: [O]) throws -> String {
    let users = messages.filter { $0["role"]?.string == "user" }
    if users.last?["content"]?.array?.contains(where: { $0["type"]?.string != "text" }) == true {
      throw ConversationFailure.invalid(
        "The selected image model does not support reference-image editing")
    }
    return users.map(ConversationDocument.text)
      .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      .filter { !$0.isEmpty }.joined(separator: "\n")
  }
  /// Inline fallback uses the web document schema (empty attachmentId plus
  /// dataUrl). A stable presentation-only ID never pretends it is stored.
  static func pictureID(_ part: O, messageID: String, index: Int) -> String? {
    guard part["type"]?.string == "image", part["gen"] != nil else { return nil }
    if let id = part["attachmentId"]?.string, ConversationDocument.validID(id) { return id }
    guard let url = part["dataUrl"]?.string, url.utf8.count <= 64 * 1024 * 1024,
      ["png", "jpeg", "webp"].contains(where: { url.hasPrefix("data:image/\($0);base64,") })
    else { return nil }
    let digest = SHA256.hash(data: Data("\(messageID):\(index)".utf8))
    return "inline-" + digest.prefix(16).map { String(format: "%02x", $0) }.joined()
  }
  static func validate(_ p: O) throws {
    func integer(_ key: String, _ range: ClosedRange<Int>) -> Bool {
      guard let v = p[key]?.double, v.isFinite, v.rounded() == v,
        let n = p[key]?.integer
      else { return false }
      return range.contains(n)
    }
    guard Set(p.keys).isSubset(of: Set(defaults.keys)),
      ["auto", "low", "medium", "high"].contains(p["quality"]?.string ?? ""),
      ["png", "webp", "jpeg"].contains(p["format"]?.string ?? ""),
      ["auto", "opaque", "transparent"].contains(p["background"]?.string ?? ""),
      integer("n", 1...4), integer("previews", 0...3),
      p["steps"] == .null || integer("steps", 1...100),
      ["thread", "random"].contains(p["seed"]?.string ?? "") || integer("seed", 0...Int(Int32.max)),
      !(p["format"]?.string == "jpeg" && p["background"]?.string == "transparent")
    else { throw ConversationFailure.invalid("Invalid image settings") }
    let size = p["size"]?.string ?? ""
    if size != "auto" {
      let pieces = size.split(separator: "x", omittingEmptySubsequences: false)
      let sides = pieces.compactMap { Int($0) }
      guard pieces.count == 2, sides.count == 2,
        sides.allSatisfy({ (32...2752).contains($0) && $0 % 32 == 0 })
      else {
        throw ConversationFailure.invalid(
          "Image dimensions must be multiples of 32, up to 2752 pixels")
      }
    }
  }
  static func seed(
    _ p: O, document: ConversationDocument, message: O, editing: Bool = false, draw: () -> Int
  ) -> Int {
    if let pinned = p["seed"]?.integer { return pinned }
    let retry = document.messages.contains {
      $0["role"]?.string == "assistant" && $0["id"] != message["id"]
        && $0["parentId"] == message["parentId"] && $0["imageGen"] != nil
        && !(message["group"] != nil && $0["group"] == message["group"])
    }
    // The reference carries the composition. Reusing its generating noise
    // can corrupt an edit; automatic edits draw afresh, as in web Studio.
    if p["seed"]?.string == "thread", !editing, !retry,
      let value = document.activeMessages.prefix(while: { $0["id"] != message["id"] })
        .reversed().compactMap({ $0["imageGen"]?["seed"]?.integer }).first
    {
      return value
    }
    if let group = message["group"]?.string {
      var hash: UInt32 = 0x811c_9dc5
      for c in group.utf16 { hash = (hash ^ UInt32(c)) &* 0x0100_0193 }
      return Int(hash & 0x7fff_ffff)
    }
    return draw()
  }
  static func body(model: String, prompt: String, params: O, seed: Int, caps: O) throws -> O {
    try validate(params)
    guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw ConversationFailure.invalid("Describe the picture to make")
    }
    let count = params["n"]?.integer ?? 1
    if let size = params["size"]?.string, size != "auto" {
      let grid = max(1, caps["size_multiple"]?.integer ?? 32)
      let limit = caps["max_side"]?.integer ?? 2752
      guard
        size.split(separator: "x").compactMap({ Int($0) }).allSatisfy({
          $0 <= limit && $0 % grid == 0
        })
      else {
        throw ConversationFailure.invalid("The selected endpoint does not support this image size")
      }
    }
    guard count <= caps["max_n"]?.integer ?? 1,
      (params["steps"]?.integer ?? 1) <= caps["max_steps"]?.integer ?? 100,
      (caps["output_formats"]?.array?.compactMap(\.string) ?? ["png"]).contains(
        params["format"]?.string ?? "")
    else {
      throw ConversationFailure.invalid(
        "The selected endpoint does not support these image settings")
    }
    let previews =
      count > 1 || caps["stream"]?.bool != true
      ? 0
      : min(params["previews"]?.integer ?? 0, caps["max_partial_images"]?.integer ?? 0)
    var body: O = [
      "model": .string(model), "prompt": .string(prompt),
      "n": .number(Decimal(count)), "seed": .number(Decimal(seed)),
      "output_format": params["format"]!,
    ]
    for name in ["size", "quality", "background"] where params[name]?.string != "auto" {
      body[name] = params[name]
    }
    if let steps = params["steps"]?.integer { body["steps"] = .number(Decimal(steps)) }
    if previews > 0 {
      body["stream"] = .bool(true)
      body["partial_images"] = .number(Decimal(previews))
    }
    return body
  }
  static func footer(_ image: O) -> String {
    guard let seconds = image["elapsedS"]?.double, seconds > 0 else { return "" }
    return NativeMessagePresentation.join([
      NativeMessagePresentation.duration(seconds * 1000),
      image["sPerStep"]?.double.map { String(format: "%.2f s/step", $0) },
    ])
  }
  static func secondsPerStep(elapsed: Double, steps: Int, images: Int) -> Double {
    elapsed / Double(max(1, steps)) / Double(max(1, images))
  }
  static func hint(_ image: O, usage: O) -> String {
    guard image["elapsedS"]?.double != nil else { return "" }
    return NativeMessagePresentation.join([
      "Render time from send to done",
      usage["ttftMs"]?.double.map {
        "\(NativeMessagePresentation.duration($0)) to the first preview"
      },
      usage["promptTokens"]?.integer.map { "\($0) prompt tokens read" },
      usage["completionTokens"]?.integer.map { "\($0) latent tokens drawn" },
    ])
  }
  static func sections(_ image: O, run: O?) -> [V] {
    let p = image["params"]?.object ?? defaults
    let prompt = image["prompt"]?.string ?? ""
    let seedPolicy =
      p["seed"]?.string == "random"
      ? "drawn for this turn"
      : p["seed"]?.string == "thread" ? "automatic" : "pinned"
    var rows: [(String, String)] = [
      ("Model", run?["model"]?.string ?? "-"),
      ("Prompt", prompt.count > 200 ? String(prompt.prefix(200)) + "..." : prompt),
      ("Seed", "\(image["seed"]?.integer ?? 0) (\(seedPolicy))"),
      (
        "Picture",
        NativeMessagePresentation.join([
          image["size"]?.string,
          image["steps"]?.integer.map { "\($0) steps" },
          p["quality"]?.string.flatMap { $0 == "auto" ? nil : "quality \($0)" },
          p["format"]?.string, p["background"]?.string.flatMap { $0 == "auto" ? nil : $0 },
        ])
      ),
    ]
    if let count = p["n"]?.integer, count > 1 {
      rows.append(("Count", "\(count) pictures on one seed"))
    }
    if let references = image["references"]?.integer, references > 0 {
      rows.append(
        (
          "References",
          "\(references) \(image["referencesFrom"]?.string == "previous" ? "from the previous picture" : "attached")"
        ))
    }
    if let previews = image["previews"]?.integer, previews > 0 {
      rows.append(("Previews", "\(previews) while rendering"))
    }
    if let seconds = image["elapsedS"]?.double {
      rows.append(
        (
          "Render",
          NativeMessagePresentation.join([
            String(format: "%.1f s", seconds),
            image["sPerStep"]?.double.map { String(format: "%.2f s/step all-in", $0) },
          ])
        ))
    }
    if run?["contended"]?.bool == true {
      rows.append(("Concurrency", "Other compare lanes shared the GPU during this run"))
    }
    return [
      .object([
        "id": .string("provenance"), "title": .string("Provenance"),
        "rows": .array(rows.map { .object(["label": .string($0.0), "value": .string($0.1)]) }),
      ])
    ]
  }
}
