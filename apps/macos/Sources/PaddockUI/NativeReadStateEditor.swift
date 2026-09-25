import AppKit
import SwiftUI

/// State-only paste interception. The questions/JSON editors, other windows,
/// text paste, IME, selection and undo keep their ordinary native behavior.
struct NativeReadStateEditor: NSViewRepresentable {
  @Binding var text: String
  let onPasteImages: (NSPasteboard) -> Bool
  @Environment(\.isEnabled) private var isEnabled

  func makeCoordinator() -> Coordinator { Coordinator(self) }
  func makeNSView(context: Context) -> NSScrollView {
    let scroll = NSScrollView(frame: NSRect(x: 0, y: 0, width: 640, height: 160))
    scroll.hasVerticalScroller = true
    PaddockScrollbars.install(on: scroll)
    let storage = NSTextContentStorage()
    let layout = NSTextLayoutManager()
    storage.addTextLayoutManager(layout)
    let container = NSTextContainer(
      size: NSSize(width: 640, height: CGFloat.greatestFiniteMagnitude))
    container.widthTracksTextView = true
    layout.textContainer = container
    let editor = ReadStateTextView(frame: scroll.bounds, textContainer: container)
    editor.contentStorage = storage
    editor.isRichText = false
    editor.allowsUndo = true
    editor.font = .systemFont(ofSize: 13)
    editor.textColor = .labelColor
    editor.backgroundColor = .textBackgroundColor
    editor.textContainerInset = NSSize(width: 0, height: 5)
    editor.isVerticallyResizable = true
    editor.isHorizontallyResizable = false
    editor.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
    editor.autoresizingMask = [.width]
    editor.delegate = context.coordinator
    editor.string = text
    editor.isEditable = isEnabled
    editor.onPasteImages = onPasteImages
    editor.setAccessibilityLabel("Text to read")
    editor.setAccessibilityIdentifier("reads-state-editor")
    context.coordinator.editor = editor
    scroll.documentView = editor
    return scroll
  }
  func updateNSView(_ scroll: NSScrollView, context: Context) {
    context.coordinator.parent = self
    guard let editor = scroll.documentView as? ReadStateTextView else { return }
    editor.onPasteImages = onPasteImages
    editor.isEditable = isEnabled
    if editor.string != text, !editor.hasMarkedText() {
      let selected = editor.selectedRange()
      editor.string = text
      editor.undoManager?.removeAllActions()
      editor.setSelectedRange(
        StudioDraftEditor.clampedSelection(selected, length: (text as NSString).length))
    }
  }
  @MainActor final class Coordinator: NSObject, NSTextViewDelegate {
    var parent: NativeReadStateEditor
    weak var editor: ReadStateTextView?
    private let edits = UndoManager()
    init(_ parent: NativeReadStateEditor) {
      self.parent = parent
      super.init()
      for name in [Notification.Name.NSUndoManagerDidUndoChange, .NSUndoManagerDidRedoChange] {
        NotificationCenter.default.addObserver(
          self, selector: #selector(historyChanged), name: name, object: edits)
      }
    }
    deinit { NotificationCenter.default.removeObserver(self) }
    func undoManager(for view: NSTextView) -> UndoManager? { edits }
    func textDidChange(_ notification: Notification) {
      if let editor { parent.text = editor.string }
    }
    @objc private func historyChanged(_ notification: Notification) {
      if let editor { parent.text = editor.string }
    }
  }
}

@MainActor final class ReadStateTextView: NSTextView {
  var contentStorage: NSTextContentStorage?
  var onPasteImages: ((NSPasteboard) -> Bool)?
  override var readablePasteboardTypes: [NSPasteboard.PasteboardType] {
    super.readablePasteboardTypes + [.png, .tiff, .init("public.jpeg"), .fileURL]
  }
  @discardableResult func acceptImages(_ board: NSPasteboard) -> Bool {
    isEditable && (onPasteImages?(board) ?? false)
  }
  override func paste(_ sender: Any?) {
    if !acceptImages(.general) { super.paste(sender) }
  }
}
