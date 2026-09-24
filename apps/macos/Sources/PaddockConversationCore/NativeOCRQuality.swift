import Foundation
import zlib

/// Same gzip ratio and 4.5 review threshold as web ocr.ts. A heuristic, not
/// proof of incorrect OCR: flags review without discarding extracted content.
enum NativeOCRQuality {
  static let threshold = 4.5
  static func repetitionRatio(_ text: String) throws -> Double {
    guard text.utf16.count >= 400 else { return 1 }
    guard text.utf8.count <= 4 * 1024 * 1024 else { throw ConversationFailure.tooLarge }
    try Task.checkCancellation()
    var stream = z_stream()
    guard
      deflateInit2_(
        &stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 31, 8,
        Z_DEFAULT_STRATEGY, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK
    else {
      throw ConversationFailure.invalid("Could not check extraction quality")
    }
    defer { deflateEnd(&stream) }
    var input = Array(text.utf8)
    var output = [UInt8](repeating: 0, count: 16 * 1024)
    let inputCount = input.count
    try input.withUnsafeMutableBytes { source in
      stream.next_in = source.bindMemory(to: Bytef.self).baseAddress
      stream.avail_in = uInt(source.count)
      var status = Z_OK
      while status != Z_STREAM_END {
        try Task.checkCancellation()
        status = output.withUnsafeMutableBytes { destination in
          stream.next_out = destination.bindMemory(to: Bytef.self).baseAddress
          stream.avail_out = uInt(destination.count)
          return deflate(&stream, Z_FINISH)
        }
        guard status == Z_OK || status == Z_STREAM_END else {
          throw ConversationFailure.invalid("Could not check extraction quality")
        }
      }
    }
    return Double(inputCount) / Double(max(1, stream.total_out))
  }
}
