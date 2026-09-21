import AppKit
import SwiftUI
import Testing

@testable import PaddockStudio
@testable import PaddockUI

@Suite("Native composer PDF selections", .serialized) @MainActor
struct StudioAttachmentTests {
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
