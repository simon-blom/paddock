import AppKit

/// Keep AppKit's selection, accessibility, IME and undo machinery. Code-specific
/// commands use the same shouldChange/replace/didChange transaction as typing.
// This storage belongs exclusively to the main-actor NSTextView; syntax workers
// receive immutable String snapshots, never the storage or its delegate.
@MainActor final class ArtifactCodeTextView: NSTextView, @preconcurrency NSTextStorageDelegate {
  var save: (() -> Void)?
  var contextAt: ((Int) -> ArtifactSourceSyntax.Context?)?
  var matchAt: ((Int) -> (Int, Int)?)?
  var completionWords: [String] = []
  var onStorageEdit: ((Int) -> Void)?
  private struct Pair {
    var open: Int
    var close: Int
    let closing: UInt16
  }
  private var insertedPairs: [Pair] = []
  private var pasting = false
  private let sourceUndoManager = UndoManager()
  override var undoManager: UndoManager? { sourceUndoManager }

  override func performKeyEquivalent(with event: NSEvent) -> Bool {
    guard window?.firstResponder === self else { return super.performKeyEquivalent(with: event) }
    let flags = event.modifierFlags.intersection([.command, .option, .control, .shift])
    let key = event.charactersIgnoringModifiers?.lowercased()
    if flags == .command || flags == [.command, .shift], key == "/" {
      toggleSourceComment(nil)
      return true
    }
    if flags == [.command, .shift], key == "\\" || key == "|" {
      jumpToMatchingBracket(nil)
      return true
    }
    if flags == [.option, .shift], event.keyCode == 125 {
      duplicateSourceLines(nil)
      return true
    }
    if flags == .control, key == " " {
      if isEditable { complete(nil) }
      return true
    }
    if flags == .option, event.keyCode == 53 {
      if isEditable { complete(nil) }
      return true
    }
    if flags == .command, key == "s" {
      if isEditable { save?() }
      return true
    }
    if flags == .command || flags == [.command, .option] || flags == [.command, .shift] {
      let key = event.charactersIgnoringModifiers?.lowercased()
      let action: NSTextFinder.Action?
      if key == "f" {
        action = flags.contains(.option) ? .showReplaceInterface : .showFindInterface
      } else if key == "g" {
        action = flags.contains(.shift) ? .previousMatch : .nextMatch
      } else {
        action = nil
      }
      if let action {
        let item = NSMenuItem()
        item.tag = action.rawValue
        performFindPanelAction(item)
        return true
      }
    }
    return super.performKeyEquivalent(with: event)
  }

  override func paste(_ sender: Any?) {
    performPlainInsertion { super.paste(sender) }
  }
  func performPlainInsertion(_ action: () -> Void) {
    let previous = pasting
    pasting = true
    defer { pasting = previous }
    action()
  }

  override func insertText(_ insertString: Any, replacementRange: NSRange) {
    guard isEditable else { return }
    let text = (insertString as? String) ?? (insertString as? NSAttributedString)?.string ?? ""
    let selection = selectedRange()
    let target = replacementRange.location == NSNotFound ? selection : replacementRange
    guard !pasting, !hasMarkedText(), selectedRanges.count == 1, target == selection,
      text.utf16.count == 1, let character = text.utf16.first
    else {
      super.insertText(insertString, replacementRange: replacementRange)
      return
    }
    if selection.length == 0,
      let index = insertedPairs.firstIndex(where: {
        $0.close == selection.location && $0.closing == character
      }),
      selection.location < (string as NSString).length,
      (string as NSString).character(at: selection.location) == character
    {
      insertedPairs.remove(at: index)
      setSelectedRange(NSRange(location: selection.location + 1, length: 0))
      return
    }
    guard let closing = ArtifactEditing.pairs[character],
      let context = contextAt?(selection.location),
      ArtifactEditing.allowsPairs(context.language), !context.isLiteral
    else {
      super.insertText(insertString, replacementRange: replacementRange)
      return
    }
    let quote = character == 34 || character == 39 || character == 96
    if context.language == "json", ![34, 91, 123].contains(character) {
      super.insertText(insertString, replacementRange: replacementRange)
      return
    }
    if character == 39, ["rust", "swift"].contains(context.language) {
      super.insertText(insertString, replacementRange: replacementRange)
      return
    }
    if ["html", "xml"].contains(context.language), !context.state.tag || !quote {
      super.insertText(insertString, replacementRange: replacementRange)
      return
    }
    if character == 96, !["javascript", "typescript", "bash", "go"].contains(context.language) {
      super.insertText(insertString, replacementRange: replacementRange)
      return
    }
    let ns = string as NSString
    if selection.length == 0 {
      // Don't turn apostrophes in words or a delimiter before an identifier into
      // surprising edits. Overtype applies only to a closer this editor inserted.
      let next = selection.location < ns.length ? ns.character(at: selection.location) : 0
      let previous = selection.location > 0 ? ns.character(at: selection.location - 1) : 0
      if (quote && (isWord(previous) || previous == 92))
        || (next != 0 && ![9, 10, 13, 32, 41, 93, 125, 44, 59, 58, 62].contains(next))
      {
        super.insertText(insertString, replacementRange: replacementRange)
        return
      }
    }
    let selected = ns.substring(with: selection)
    let replacement = text + selected + String(decoding: [closing], as: UTF16.self)
    if applyEdits(
      [.init(range: selection, text: replacement)], name: "Insert Delimiters",
      selections: [NSRange(location: selection.location + 1, length: selection.length)])
    {
      insertedPairs.append(
        Pair(
          open: selection.location, close: selection.location + 1 + selection.length,
          closing: closing))
    }
  }

  override func deleteBackward(_ sender: Any?) {
    guard isEditable, !hasMarkedText(), selectedRanges.count == 1 else {
      super.deleteBackward(sender)
      return
    }
    let selection = selectedRange()
    if selection.length == 0,
      insertedPairs.contains(where: {
        $0.open == selection.location - 1 && $0.close == selection.location
      })
    {
      _ = applyEdits(
        [.init(range: NSRange(location: selection.location - 1, length: 2), text: "")],
        name: "Delete Delimiters",
        selections: [NSRange(location: selection.location - 1, length: 0)])
    } else {
      super.deleteBackward(sender)
    }
  }

  func textStorage(
    _ textStorage: NSTextStorage, willProcessEditing editedMask: NSTextStorageEditActions,
    range editedRange: NSRange, changeInLength delta: Int
  ) {
    guard editedMask.contains(.editedCharacters) else { return }
    onStorageEdit?(editedRange.location)
    if undoManager?.isUndoing == true || undoManager?.isRedoing == true {
      insertedPairs.removeAll()
      return
    }
    // willProcess is intentional: attribute fixing can widen didProcess's range
    // to the whole paragraph, which would incorrectly discard untouched pairs.
    let old = NSRange(location: editedRange.location, length: max(0, editedRange.length - delta))
    insertedPairs = insertedPairs.compactMap { pair in
      if old.contains(pair.open) || old.contains(pair.close) { return nil }
      var next = pair
      if old.location <= next.open { next.open += delta }
      if old.location <= next.close { next.close += delta }
      return next.open >= 0 && next.close > next.open ? next : nil
    }
  }

  func resetEditingState() {
    insertedPairs.removeAll()
    completionWords.removeAll()
  }
  private func isWord(_ c: UInt16) -> Bool {
    (48...57).contains(c) || (65...90).contains(c) || (97...122).contains(c) || c == 95 || c >= 128
  }

  override func insertTab(_ sender: Any?) {
    guard isEditable, !hasMarkedText() else {
      super.insertTab(sender)
      return
    }
    if selectedRange().length == 0, selectedRanges.count == 1 {
      let ns = string as NSString
      let position = selectedRange().location
      let start = ns.lineRange(for: NSRange(location: position, length: 0)).location
      let before = ns.substring(with: NSRange(location: start, length: position - start))
      let spaces = 2 - before.reduce(0, { $0 + ($1 == "\t" ? 2 - $0 % 2 : 1) }) % 2
      super.insertText(String(repeating: " ", count: spaces), replacementRange: selectedRange())
    } else {
      indent(outdent: false)
    }
  }
  override func insertBacktab(_ sender: Any?) {
    guard isEditable, !hasMarkedText() else {
      super.insertBacktab(sender)
      return
    }
    indent(outdent: true)
  }
  override func insertNewline(_ sender: Any?) {
    guard isEditable, !hasMarkedText(), selectedRanges.count == 1 else {
      super.insertNewline(sender)
      return
    }
    let ns = string as NSString
    let selection = selectedRange()
    let range = ns.lineRange(for: NSRange(location: selection.location, length: 0))
    let before = ns.substring(
      with: NSRange(location: range.location, length: selection.location - range.location))
    let prefix = String(before.prefix { $0 == " " || $0 == "\t" })
    let trimmed = before.trimmingCharacters(in: .whitespaces)
    let context = contextAt?(selection.location)
    let unit = prefix.contains("\t") ? "\t" : "  "
    let opens =
      trimmed.last.map { "{[(".contains($0) || $0 == ":" && context?.language == "python" } == true
    let extra =
      opens && context?.isLiteral == false && ArtifactEditing.allowsPairs(context?.language ?? "")
      ? unit : ""
    let newline =
      ns.substring(with: range).hasSuffix("\r\n")
        || ns.substring(to: min(ns.length, 4096)).contains("\r\n") ? "\r\n" : "\n"
    let next = selection.location < ns.length ? ns.character(at: selection.location) : 0
    let open = trimmed.utf16.last ?? 0
    let paired = selection.length == 0 && !extra.isEmpty && ArtifactEditing.pairs[open] == next
    let inserted = newline + prefix + extra + (paired ? newline + prefix : "")
    _ = applyEdits(
      [.init(range: selection, text: inserted)], name: "Insert Newline",
      selections: [
        NSRange(location: selection.location + (newline + prefix + extra).utf16.count, length: 0)
      ])
  }

  private func indent(outdent: Bool) {
    let ns = string as NSString
    let selections = selectedRanges.map(\.rangeValue)
    var starts = Set<Int>()
    for selection in selections {
      // A selection ending at the next line's start does not include that line.
      let end = selection.length > 0 ? NSMaxRange(selection) - 1 : selection.location
      var position = ns.lineRange(for: NSRange(location: selection.location, length: 0)).location
      repeat {
        starts.insert(position)
        let next = NSMaxRange(ns.lineRange(for: NSRange(location: position, length: 0)))
        if next <= position || next > end { break }
        position = next
      } while position <= ns.length
    }
    let edits: [(NSRange, String)] = starts.sorted().compactMap { start in
      if !outdent { return (NSRange(location: start, length: 0), "  ") }
      var count = 0
      if start < ns.length, ns.character(at: start) == 9 {
        count = 1
      } else {
        while count < 2, start + count < ns.length, ns.character(at: start + count) == 32 {
          count += 1
        }
      }
      return count == 0 ? nil : (NSRange(location: start, length: count), "")
    }
    _ = applyEdits(
      edits.map { .init(range: $0.0, text: $0.1) }, name: outdent ? "Outdent" : "Indent")
  }

  @discardableResult func applyEdits(
    _ edits: [ArtifactEditing.Edit], name: String, selections: [NSRange]? = nil
  ) -> Bool {
    guard isEditable, !hasMarkedText(), !edits.isEmpty else { return false }
    breakUndoCoalescing()
    let manager = undoManager
    let automatic = manager?.groupsByEvent ?? true
    // AppKit's shouldChangeText registers the inverse. It must run inside our
    // transaction. Otherwise two commands in one event undo together, even if
    // replaceCharacters itself was enclosed in begin/endUndoGrouping.
    if automatic, manager?.groupingLevel == 1 { manager?.endUndoGrouping() }
    manager?.groupsByEvent = false
    manager?.beginUndoGrouping()
    defer {
      manager?.endUndoGrouping()
      manager?.groupsByEvent = automatic
      breakUndoCoalescing()
    }
    guard
      shouldChangeText(
        inRanges: edits.map { NSValue(range: $0.range) }, replacementStrings: edits.map(\.text))
    else { return false }
    let oldSelections = selectedRanges.map(\.rangeValue)
    for edit in edits.sorted(by: { $0.range.location > $1.range.location }) {
      replaceCharacters(in: edit.range, with: edit.text)
    }
    didChangeText()
    manager?.setActionName(name)
    selectedRanges =
      (selections
      ?? oldSelections.map {
        let start = ArtifactEditing.translated($0.location, through: edits)
        let end = ArtifactEditing.translated(NSMaxRange($0), through: edits)
        return NSRange(location: start, length: max(0, end - start))
      }).map { NSValue(range: $0) }
    return true
  }

  @objc func toggleSourceComment(_ sender: Any?) {
    guard isEditable, !hasMarkedText(), let context = contextAt?(selectedRange().location),
      let style = ArtifactEditing.comment(for: context.language)
    else { return }
    let edits = ArtifactEditing.comments(
      string as NSString, selections: selectedRanges.map(\.rangeValue), style: style)
    let selection: [NSRange]?
    if case .block = style, let edit = edits.first {
      selection = [NSRange(location: edit.range.location, length: edit.text.utf16.count)]
    } else {
      selection = nil
    }
    _ = applyEdits(edits, name: "Toggle Comment", selections: selection)
  }
  @objc func duplicateSourceLines(_ sender: Any?) {
    guard isEditable, !hasMarkedText(), selectedRanges.count == 1 else { return }
    let ns = string as NSString
    let lines = ArtifactEditing.lineRanges(ns, selections: [selectedRange()])
    guard let first = lines.first, let last = lines.last else { return }
    let range = NSRange(location: first.location, length: NSMaxRange(last) - first.location)
    let value = ns.substring(with: range)
    let newline = ns.substring(to: min(ns.length, 4096)).contains("\r\n") ? "\r\n" : "\n"
    let needsNewline = !value.hasSuffix("\n") && !value.hasSuffix("\r")
    let inserted = (needsNewline ? newline : "") + value
    let newStart = NSMaxRange(range) + (needsNewline ? newline.utf16.count : 0)
    let selection = selectedRange()
    _ = applyEdits(
      [.init(range: NSRange(location: NSMaxRange(range), length: 0), text: inserted)],
      name: "Duplicate Lines",
      selections: [
        NSRange(location: newStart + selection.location - range.location, length: selection.length)
      ])
  }
  @objc func jumpToMatchingBracket(_ sender: Any?) {
    guard let pair = matchAt?(selectedRange().location) else { return }
    setSelectedRange(NSRange(location: pair.1, length: 0))
    scrollRangeToVisible(selectedRange())
  }

  override func completions(
    forPartialWordRange charRange: NSRange, indexOfSelectedItem index: UnsafeMutablePointer<Int>
  ) -> [String]? {
    guard isEditable, !hasMarkedText(), charRange.location != NSNotFound, charRange.length > 0,
      NSMaxRange(charRange) <= (string as NSString).length,
      let context = contextAt?(charRange.location), !context.isLiteral
    else { return nil }
    let prefix = (string as NSString).substring(with: charRange)
    // Sorted vocabulary allows prefix lookup without scanning all words on each
    // keystroke. Suggestions are document words, not invented API semantics.
    var low = 0
    var high = completionWords.count
    while low < high {
      let mid = (low + high) / 2
      if completionWords[mid] < prefix { low = mid + 1 } else { high = mid }
    }
    var result: [String] = []
    while low < completionWords.count, result.count < 100, completionWords[low].hasPrefix(prefix) {
      if completionWords[low] != prefix { result.append(completionWords[low]) }
      low += 1
    }
    index.pointee = 0
    return result
  }

  override func menu(for event: NSEvent) -> NSMenu? {
    let menu = (super.menu(for: event)?.copy() as? NSMenu) ?? NSMenu()
    menu.addItem(.separator())
    for (title, action, key, modifiers) in [
      ("Toggle Comment", #selector(toggleSourceComment(_:)), "/", NSEvent.ModifierFlags.command),
      ("Duplicate Lines", #selector(duplicateSourceLines(_:)), "\u{F701}", [.option, .shift]),
      ("Jump to Matching Bracket", #selector(jumpToMatchingBracket(_:)), "\\", [.command, .shift]),
      ("Complete Word", #selector(complete(_:)), "\u{1b}", .option),
    ] {
      let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
      item.target = self
      item.keyEquivalentModifierMask = modifiers
      menu.addItem(item)
    }
    return menu
  }
  override func validateMenuItem(_ menuItem: NSMenuItem) -> Bool {
    switch menuItem.action {
    case #selector(toggleSourceComment(_:)):
      return isEditable && !hasMarkedText()
        && contextAt?(selectedRange().location).flatMap {
          ArtifactEditing.comment(for: $0.language)
        } != nil
    case #selector(duplicateSourceLines(_:)), #selector(complete(_:)):
      return isEditable && !hasMarkedText()
    case #selector(jumpToMatchingBracket(_:)): return matchAt?(selectedRange().location) != nil
    default: return super.validateMenuItem(menuItem)
    }
  }
}
