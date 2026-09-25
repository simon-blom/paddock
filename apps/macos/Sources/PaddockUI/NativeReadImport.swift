import Foundation
import UniformTypeIdentifiers

/// One native file picker serves all three actions; its purpose is captured
/// before asynchronous importing so a later action cannot reroute a selection.
enum NativeReadImport {
  case state, questions, images
  var contentTypes: [UTType] {
    switch self {
    case .state: [.item]
    case .questions: [.json]
    case .images: [.image]
    }
  }
  var allowsMultipleSelection: Bool { self == .images }
}

extension NativeReadsModel {
  func importSelection(_ result: Result<[URL], Error>, kind: NativeReadImport) async {
    switch result {
    case .success(let urls):
      guard !urls.isEmpty else { return }
      guard !historyNavigationBlocked else {
        importFailure("Wait for the current operation before importing a file.", kind: kind)
        return
      }
      if kind == .questions { questionsError = nil } else { stateError = nil }
      if kind == .images {
        await addPictures(urls)
      } else if urls.count == 1 {
        await loadFile(urls[0], asJSON: kind == .questions)
      } else {
        importFailure("Choose one file.", kind: kind)
      }
    case .failure(let error):
      let cocoa = error as NSError
      guard cocoa.domain != NSCocoaErrorDomain || cocoa.code != NSUserCancelledError else { return }
      importFailure(error.localizedDescription, kind: kind)
    }
  }
  private func importFailure(_ message: String, kind: NativeReadImport) {
    if kind == .questions { questionsError = message } else { stateError = message }
  }
}
