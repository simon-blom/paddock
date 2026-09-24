import AppKit
import SwiftUI
import Testing

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native composer PDF selections", .serialized) @MainActor
struct StudioAttachmentTests {
  private func imageCapabilities() throws -> StudioState.Capabilities {
    let raw: [String: Any] = [
      "reasoning": "none", "levels": [], "reasoningDefault": "", "reasoningOff": false,
      "preserveThinking": false, "thinkingBudget": false, "webSearch": false,
      "vision": true, "context": 8192, "ocrModes": [], "docParser": false, "pdfRaster": true,
      "imageLanes": [
        [
          "id": "bonsai", "name": "Bonsai", "context": 8192,
          "budget": [
            "max_pixels": 16_777_216, "min_pixels": 65536, "pixels_per_token": 1024,
            "max_tokens": 16384, "min_tokens": 64, "auto_max_tokens": 4096,
          ],
        ],
        [
          "id": "small", "name": "Small tower", "context": 16384,
          "budget": [
            "max_pixels": 250880, "min_pixels": 250880, "pixels_per_token": 896,
            "max_tokens": 280, "min_tokens": 280, "auto_max_tokens": 280,
          ],
        ],
        ["id": "cloud", "name": "Cloud", "context": 131072, "budget": NSNull()],
      ],
    ]
    return try JSONDecoder().decode(
      StudioState.Capabilities.self, from: JSONSerialization.data(withJSONObject: raw))
  }
  @Test func imageChoicesUseEveryCompareLanesOwnBudgetWithoutGuessingCloudCosts() throws {
    let caps = try imageCapabilities()
    var file = StudioAttachment(
      id: "photo", name: "Photo.jpg", mime: "image/jpeg", size: 18_171_177, phase: "Ready")
    #expect(caps.imageEstimates(for: file).isEmpty)
    file.width = 6720
    file.height = 4480
    let automatic = caps.imageEstimates(for: file)
    #expect(automatic.map(\.tokens) == [4096, 280])
    #expect(!automatic.contains { $0.exceedsContext })
    let original = caps.imageEstimates(for: file, detail: "high")
    #expect(original.filter(\.exceedsContext).map(\.modelID) == ["bonsai"])
    #expect(original.map(\.tokens) == [16381, 280])
    #expect(StudioImageEstimate.label([]) == nil)
    #expect(StudioImageEstimate.label(original)?.contains("≈") == true)
    #expect(file.detail == "auto")
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: StudioAttachmentOptions(
          attachment: .constant(file), canRasterPDF: true, capabilities: caps, onDone: {}
        ).preferredColorScheme(dark ? .dark : .light))
      host.view.layoutSubtreeIfNeeded()
      let size = host.sizeThatFits(in: CGSize(width: 290, height: 1000))
      #expect(size.width == 290 && size.height < 420)
    }
  }
  @Test func photosDefaultToAutoResizeWithoutChangingTheOriginal() {
    var file = StudioAttachment(
      id: "photo", name: "Photo.jpg", mime: "image/jpeg", size: 18_171_177, phase: "Ready")
    file.width = 6720
    file.height = 4480
    #expect(file.choices.object?["detail"] == .string("auto"))
    #expect(StudioAttachment.imageDetailOptions.map(\.value) == ["auto", "high", "low"])
    #expect(
      StudioAttachment.imageDetailOptions.map(\.title) == ["Auto-resize", "Original", "Smaller"])
    file.detail = "high"
    #expect(file.choices.object?["detail"] == .string("high"))
    #expect(file.width == 6720 && file.height == 4480 && file.size == 18_171_177)
    #expect(file.choices.object?["id"] == .string("photo"))
  }
  @Test func parserPagesUseSeparateContextBudgetsButOrdinaryVisionSumsImages() throws {
    func lane(_ options: String) throws -> StudioState.Capabilities.ImageLane {
      try JSONDecoder().decode(
        StudioState.Capabilities.ImageLane.self,
        from: Data("{\"id\":\"model\",\"name\":\"Model\",\"context\":8192,\(options)}".utf8))
    }
    #expect(
      try lane("\"documentParser\":true").tokensForTurn([5000, 5000], instruction: "Read") == 5000)
    #expect(
      try lane("\"documentParser\":false").tokensForTurn([5000, 5000], instruction: "Read") == 10000
    )
    #expect(
      try lane("\"taskTags\":[\"<ocr>\"]").tokensForTurn([5000, 5000], instruction: " <ocr> ")
        == 5000)
    #expect(
      try lane("\"taskTags\":[\"<ocr>\"]").tokensForTurn([5000, 5000], instruction: "Explain")
        == 10000)
  }
  @Test func documentReadingChipsFitBothAppearancesWithoutInventingModes() {
    #expect(StudioDocumentReadingControls.label("free") == "Plain text")
    #expect(StudioDocumentReadingControls.label("spotting") == "Text spotting")
    #expect(StudioDocumentReadingControls.label("custom") == "Custom")
    for dark in [false, true] {
      let host = NSHostingController(
        rootView: StudioDocumentReadingControls(
          modes: ["ocr", "table", "chart", "formula", "spotting", "seal"], grounding: true,
          mode: "ocr", regions: true, onMode: { _ in }, onRegions: { _ in }
        ).preferredColorScheme(dark ? .dark : .light).frame(width: 320))
      host.view.layoutSubtreeIfNeeded()
      let size = host.sizeThatFits(in: CGSize(width: 320, height: 1000))
      #expect(size.width == 320 && size.height > 20 && size.height < 64)
    }
  }
  private func pdf() -> StudioAttachment {
    var file = StudioAttachment(
      id: "pdf", name: "Annual report.pdf", mime: "application/pdf", size: 8192, phase: "Ready")
    file.pages = 12
    return file
  }
  @Test func rangesAreVisibleInclusiveAndPerAttachment() {
    var file = pdf()
    #expect(file.pageSummary == "All 12 pages")
    #expect(file.allPages)
    file.from = 2
    file.to = 4
    file.textOnly = true
    #expect(file.pageSummary == "Pages 2-4")
    #expect(file.choices.object?["from"] == .number(2))
    #expect(file.choices.object?["to"] == .number(4))
    #expect(file.choices.object?["text"] == .bool(true))
    #expect(pdf().allPages)
    file.to = 2
    #expect(file.pageSummary == "Page 2")
    file.to = nil
    #expect(file.pageSummary == "Pages 2-12")
    file.pages = nil
    #expect(file.pageSummary == "Pages 2-end")
    file.from = nil
    #expect(file.pageSummary == "All pages")
    #expect(file.choices.object?["from"] == nil)
    #expect(file.choices.object?["to"] == nil)
  }
  @Test func invalidEditsCannotSilentlyUseThePreviousNumber() {
    for raw in [
      "abc", "2-4", "1.5", "-2", "0", "13", "1e2", "9007199254740992",
      String(repeating: "9", count: 40),
    ] {
      var file = pdf()
      file.from = 2
      file.firstPage = raw
      #expect(file.selectionError != nil)
      #expect(file.pageSummary == "Check page range")
    }
    var file = pdf()
    file.firstPage = " 2 "
    file.lastPage = "4"
    #expect(file.selectionError == nil)
    file.lastPage = "1"
    #expect(file.selectionError == "The page range ends before it starts.")
    file.firstPage = ""
    file.lastPage = "13"
    #expect(file.selectionError != nil)
    file.lastPage = " "
    #expect(file.allPages)
    #expect(file.selectionError == nil)
  }
  @Test func mimeDetectionAndUnknownCountRemainHonest() {
    let file = StudioAttachment(
      id: "pdf", name: "document", mime: "application/pdf", size: 20, phase: "Ready")
    #expect(file.isPDF && file.supportsPageSelection)
    #expect(file.pageSummary == "All pages")
    var single = pdf()
    single.pages = 1
    #expect(single.pageSummary == "All 1 page")
    single.from = 2
    #expect(single.selectionError != nil)
  }
  @Test func optionsFitInBothAppearancesAndWithUnknownCounts() {
    _ = NSApplication.shared
    for dark in [false, true] {
      for count: Int? in [12, nil] {
        var file = pdf()
        file.pages = count
        let host = NSHostingController(
          rootView: StudioAttachmentOptions(
            attachment: .constant(file), canRasterPDF: true, onDone: {}
          )
          .preferredColorScheme(dark ? .dark : .light))
        host.view.layoutSubtreeIfNeeded()
        let size = host.sizeThatFits(in: CGSize(width: 290, height: 1000))
        #expect(size.width == 290)
        #expect(size.height > 200 && size.height < 360)
      }
    }
  }
}
