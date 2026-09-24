import AppKit

/// Rendering-only marks: no source edits, no injected HTML, no text-storage
/// replacement and no ownership of the library's delegate. Copy and selection
/// keep the exact extraction. Matching is off-main and cancellation/versioned.
@MainActor final class NativeConfidenceDecoration {
  var words: [String] = [] {
    didSet {
      guard words != oldValue else { return }
      for entry in entries.values { entry.update(words) }
    }
  }
  private var entries: [ObjectIdentifier: Entry] = [:]
  func reconcile(_ root: NSView) {
    guard !words.isEmpty || !entries.isEmpty else { return }
    var live = Set<ObjectIdentifier>()
    func visit(_ view: NSView) {
      if let text = view as? NSTextView {
        let id = ObjectIdentifier(text)
        live.insert(id)
        if entries[id] == nil { entries[id] = Entry(text, words: words) }
      }
      for child in view.subviews { visit(child) }
    }
    visit(root)
    entries = entries.filter { live.contains($0.key) }
  }

  @MainActor private final class Entry {
    weak var view: NSTextView?
    var words: [String]
    var task: Task<Void, Never>?
    // NotificationCenter permits removal on any thread (including deinit on
    // macOS 15.0); callback execution and all text-view access stay main-actor.
    nonisolated(unsafe) var observer: (any NSObjectProtocol)?
    var ranges: [NSRange] = []
    init(_ view: NSTextView, words: [String]) {
      self.view = view
      self.words = words
      if let storage = view.textStorage {
        observer = NotificationCenter.default.addObserver(
          forName: NSTextStorage.didProcessEditingNotification, object: storage, queue: .main
        ) { [weak self] _ in
          MainActor.assumeIsolated { self?.refresh() }
        }
      }
      refresh()
    }
    deinit {
      task?.cancel()
      if let observer { NotificationCenter.default.removeObserver(observer) }
    }
    func update(_ value: [String]) {
      words = value
      refresh()
    }
    func refresh() {
      task?.cancel()
      task = Task { [weak self] in
        // Storage notifications can arrive inside a text-storage transaction.
        await Task.yield()
        guard !Task.isCancelled, let self, let view, let storage = view.textStorage else { return }
        let text = storage.string
        var excluded: [NSRange] = []
        storage.enumerateAttributes(in: NSRange(location: 0, length: storage.length)) {
          attrs, range, _ in
          let font = attrs[.font] as? NSFont
          if attrs[.backgroundColor] != nil || attrs[.attachment] != nil
            || font?.fontDescriptor.symbolicTraits.contains(.monoSpace) == true
          {
            excluded.append(range)
          }
        }
        let terms = words
        let blocked = excluded
        let job = Task.detached(priority: .utility) {
          NativeConfidenceMatches.ranges(text: text, words: terms, excluding: blocked)
        }
        let next = await withTaskCancellationHandler {
          await job.value
        } onCancel: {
          job.cancel()
        }
        guard !Task.isCancelled, view.string == text else { return }
        apply(next, to: view)
      }
    }
    private func apply(_ next: [NSRange], to view: NSTextView) {
      // A single caution band, matching web's unsure-word semantics, while
      // respecting native light/dark appearance and accessibility selection.
      let color = NSColor.systemYellow.withAlphaComponent(0.22)
      if let layout = view.textLayoutManager, let content = layout.textContentManager {
        func textRange(_ range: NSRange) -> NSTextRange? {
          guard
            let start = content.location(content.documentRange.location, offsetBy: range.location),
            let end = content.location(start, offsetBy: range.length)
          else { return nil }
          return NSTextRange(location: start, end: end)
        }
        for old in ranges {
          if let range = textRange(old) {
            layout.removeRenderingAttribute(.backgroundColor, for: range)
          }
        }
        for mark in next {
          if let range = textRange(mark) {
            layout.addRenderingAttribute(.backgroundColor, value: color, for: range)
          }
        }
      } else if let layout = view.layoutManager {
        let length = (view.string as NSString).length
        for old in ranges where NSMaxRange(old) <= length {
          layout.removeTemporaryAttribute(.backgroundColor, forCharacterRange: old)
        }
        for mark in next {
          layout.addTemporaryAttribute(.backgroundColor, value: color, forCharacterRange: mark)
        }
      }
      ranges = next
      view.needsDisplay = true
    }
  }
}

enum NativeConfidenceMatches {
  static func ranges(text: String, words: [String], excluding: [NSRange] = []) -> [NSRange] {
    guard text.utf8.count <= 4 * 1024 * 1024 else { return [] }
    let terms = Set(words.filter { $0.count > 1 && $0.utf8.count <= 256 }).sorted {
      $0.utf8.count == $1.utf8.count ? $0 < $1 : $0.utf8.count > $1.utf8.count
    }.prefix(512)
    guard !terms.isEmpty,
      let regex = try? NSRegularExpression(
        pattern: terms.map(NSRegularExpression.escapedPattern).joined(separator: "|"))
    else { return [] }
    var result: [NSRange] = []
    let blocked = excluding.sorted { $0.location < $1.location }
    var block = 0
    regex.enumerateMatches(in: text, range: NSRange(text.startIndex..., in: text)) {
      match, _, stop in
      if Task.isCancelled || result.count >= 4096 {
        stop.pointee = true
        return
      }
      guard let range = match?.range else { return }
      while block < blocked.count && NSMaxRange(blocked[block]) <= range.location { block += 1 }
      if block < blocked.count && NSIntersectionRange(blocked[block], range).length > 0 { return }
      result.append(range)
    }
    return result
  }
}
