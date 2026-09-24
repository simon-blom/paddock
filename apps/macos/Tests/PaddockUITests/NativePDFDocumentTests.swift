import CoreGraphics
import Foundation
import ImageIO
import PaddockConversationCore
import Testing

@testable import PaddockStudio

struct NativePDFDocumentTests {
  func fixture(width: Double = 600, height: Double = 800) throws -> Data {
    let bytes = NSMutableData()
    let consumer = try #require(CGDataConsumer(data: bytes))
    var box = CGRect(x: 0, y: 0, width: width, height: height)
    let context = try #require(CGContext(consumer: consumer, mediaBox: &box, nil))
    for _ in 0..<200 {
      context.beginPDFPage(nil)
      context.setFillColor(CGColor(gray: 0, alpha: 1))
      context.fill(CGRect(x: 100, y: 200, width: 100, height: 100))
      context.endPDFPage()
    }
    context.closePDF()
    return bytes as Data
  }
  @Test func largePDFIsRetainedCompressedAndOnlyRequestedPagesRasterize() async throws {
    let bytes = try fixture()
    let source = try await NativePDFInput.shared.source(bytes)
    #expect(source.pageCount == 200)
    for page in [1, 197, 200] {
      let raster = try await source.render(page, 1_500_000)
      #expect(raster.width * raster.height <= 1_500_000)
      #expect(abs(Double(raster.width) / Double(raster.height) - 0.75) < 0.001)
      let uri = try #require(raster.image["image_url"]?.string)
      let data = try #require(
        Data(base64Encoded: String(uri.dropFirst("data:image/jpeg;base64,".count))))
      let image = try #require(CGImageSourceCreateWithData(data as CFData, nil))
      let decoded = try #require(CGImageSourceCreateImageAtIndex(image, 0, nil))
      #expect(decoded.width == raster.width && decoded.height == raster.height)
    }
    await #expect(throws: ConversationFailure.self) { _ = try await source.render(201, 1_500_000) }
    await #expect(throws: ConversationFailure.self) { _ = try await source.render(1, Int.max) }
    await #expect(throws: ConversationFailure.self) {
      _ = try await NativePDFInput.shared.source(Data("invalid".utf8))
    }
  }
  @Test func pathologicalAspectRatiosCannotExceedThePixelBudget() async throws {
    for size in [(1_000_000.0, 1.0), (1.0, 1_000_000.0)] {
      let source = try await NativePDFInput.shared.source(fixture(width: size.0, height: size.1))
      let raster = try await source.render(1, 4)
      #expect(raster.width > 0 && raster.height > 0 && raster.width * raster.height <= 4)
    }
  }
}
