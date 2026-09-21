import Foundation

/// Native port of studio/src/lib/tree.ts. Compare lanes form one step; the
/// first lane anchors children. Index once per operation, not per message.
public struct ConversationStep: Sendable, Equatable {
  public let messages: [ConversationDocument.Object]
  public var anchorID: String { messages[0]["id"]!.string! }
  public var parentID: String { messages[0]["parentId"]?.string ?? "" }
  public var comparison: Bool {
    messages[0]["role"]?.string == "assistant" && !(messages[0]["group"]?.string ?? "").isEmpty
  }
}

private struct TreeIndex {
  let messages: [ConversationDocument.Object]
  let byID: [String: Int]
  let children: [String: [Int]]
  init(_ doc: ConversationDocument) {
    messages = doc.messages
    var ids = [String: Int]()
    var children = [String: [Int]]()
    for (i, m) in messages.enumerated() {
      ids[m["id"]!.string!] = i
      children[m["parentId"]?.string ?? "", default: []].append(i)
    }
    byID = ids
    self.children = children
  }
  func steps(_ children: [Int]) -> [ConversationStep] {
    var groups = [String: Int]()
    var steps = [[ConversationDocument.Object]]()
    for i in children {
      let m = messages[i]
      if m["role"]?.string == "assistant", let group = m["group"]?.string, !group.isEmpty {
        if let at = groups[group] {
          steps[at].append(m)
          continue
        }
        groups[group] = steps.count
      }
      steps.append([m])
    }
    return steps.map { ConversationStep(messages: $0) }
  }
  func active(_ leaf: String?) -> [ConversationStep] {
    var cursor = leaf
    var seen = Set<String>()
    var chain = [Int]()
    while let id = cursor, let at = byID[id], seen.insert(id).inserted {
      chain.append(at)
      cursor = messages[at]["parentId"]?.string
    }
    return chain.reversed().map { at in
      let m = messages[at]
      if m["role"]?.string == "assistant", let group = m["group"]?.string, !group.isEmpty {
        return ConversationStep(
          messages: (children[m["parentId"]?.string ?? ""] ?? [])
            .map { messages[$0] }.filter { $0["group"]?.string == group })
      }
      return ConversationStep(messages: [m])
    }
  }
  func descend(_ anchor: String, memory: [String: String]) -> String {
    var cur = anchor
    var seen = Set<String>()
    while seen.insert(cur).inserted {
      let steps = steps(children[cur] ?? [])
      guard let last = steps.last else { break }
      cur = steps.first { $0.anchorID == memory[cur] }?.anchorID ?? last.anchorID
    }
    return cur
  }
}

extension ConversationDocument {
  public var messageControls: [String: ConversationMessageControls] {
    let idx = TreeIndex(self)
    let path = idx.active(leafID).flatMap(\.messages)
    var branches = [String: ConversationMessageControls.Branch]()
    for children in idx.children.values {
      let steps = idx.steps(children)
      guard steps.count > 1 else { continue }
      for (i, step) in steps.enumerated() {
        branches[step.anchorID] = .init(
          index: i + 1, count: steps.count,
          previous: i > 0 ? steps[i - 1].anchorID : nil,
          next: i + 1 < steps.count ? steps[i + 1].anchorID : nil)
      }
    }
    let lastID = path.last?["id"]?.string
    let compare = !(fields["compareModels"]?.array ?? []).isEmpty
    return Dictionary(
      uniqueKeysWithValues: path.map { message in
        let id = message["id"]!.string!
        let role = message["role"]?.string
        let retry =
          id == lastID && role == "assistant" && (message["group"]?.string ?? "").isEmpty
          && !compare
        return (
          id,
          .init(
            edit: role == "user" && message["auto"]?.bool != true, retry: retry,
            continueReply: retry && message["incomplete"]?.string == "length", branch: branches[id])
        )
      })
  }
  public var activeSteps: [ConversationStep] { TreeIndex(self).active(leafID) }
  public var activeMessages: [Object] { activeSteps.flatMap(\.messages) }
  public var tipID: String? { activeSteps.last?.anchorID }

  /// An in-memory migration on load, never an unsolicited database rewrite.
  @discardableResult public mutating func migrateTree() -> Bool {
    var messages = messages
    var changed = false
    if messages.isEmpty {
      if fields["leafId"] != nil {
        setLeaf(nil)
        return true
      }
      return false
    }
    let present = Set(messages.map { $0["id"]!.string! })
    var previous: String?
    var i = 0
    while i < messages.count {
      let first = i
      let message = messages[first]
      if message["role"]?.string == "assistant", let group = message["group"]?.string,
        !group.isEmpty
      {
        while i + 1 < messages.count, messages[i + 1]["role"]?.string == "assistant",
          messages[i + 1]["group"]?.string == group
        { i += 1 }
      }
      for j in first...i {
        let parent = messages[j]["parentId"]
        let broken = parent?.string.map { !present.contains($0) } ?? false
        if parent == nil || broken {
          messages[j]["parentId"] = previous.map(Value.string) ?? .null
          changed = true
        }
      }
      previous = messages[first]["id"]!.string!
      i += 1
    }
    if changed { replaceMessages(messages) }
    if leafID == nil || !present.contains(leafID!) {
      setLeaf(previous)
      changed = true
    }
    if fields["branchMemory"] != nil {
      let cleaned = branchMemory.filter {
        ($0.key.isEmpty || present.contains($0.key)) && present.contains($0.value)
      }
      if cleaned != branchMemory {
        setMemory(cleaned)
        changed = true
      }
    }
    return changed
  }

  public func siblings(of id: String) -> (steps: [ConversationStep], index: Int)? {
    let idx = TreeIndex(self)
    guard let at = idx.byID[id] else { return nil }
    let steps = idx.steps(idx.children[idx.messages[at]["parentId"]?.string ?? ""] ?? [])
    guard let i = steps.firstIndex(where: { $0.messages.contains { $0["id"]?.string == id } })
    else { return nil }
    return (steps, i)
  }
  @discardableResult public mutating func stepSibling(of id: String, delta: Int) -> Bool {
    guard delta == -1 || delta == 1, let info = siblings(of: id), info.steps.count > 1,
      info.steps.indices.contains(info.index + delta)
    else { return false }
    focusStep(info.steps[info.index + delta].anchorID)
    return true
  }
  public mutating func focusStep(_ anchorID: String) {
    let idx = TreeIndex(self)
    guard let at = idx.byID[anchorID] else { return }
    var memory = branchMemory
    for step in idx.active(leafID) { memory[step.parentID] = step.anchorID }
    memory[idx.messages[at]["parentId"]?.string ?? ""] = anchorID
    setMemory(memory)
    setLeaf(idx.descend(anchorID, memory: memory))
  }
  /// Only mutates the supplied value; callers must confirm/commit deletion.
  @discardableResult public mutating func deleteSubtree(_ id: String) -> Set<String> {
    let idx = TreeIndex(self)
    guard let at = idx.byID[id], let info = siblings(of: id) else { return [] }
    var stack = info.steps[info.index].messages.map { $0["id"]!.string! }
    var removed = Set<String>()
    while let current = stack.popLast() {
      guard removed.insert(current).inserted else { continue }
      stack.append(
        contentsOf: (idx.children[current] ?? []).map { idx.messages[$0]["id"]!.string! })
    }
    let parent = idx.messages[at]["parentId"]?.string
    replaceMessages(idx.messages.filter { !removed.contains($0["id"]!.string!) })
    let alive = Set(messages.map { $0["id"]!.string! })
    if fields["branchMemory"] != nil {
      setMemory(
        branchMemory.filter {
          ($0.key.isEmpty || alive.contains($0.key)) && alive.contains($0.value)
        })
    }
    let next = TreeIndex(self)
    let survivors = next.steps(next.children[parent ?? ""] ?? [])
    if let last = survivors.last {
      setLeaf(next.descend(last.anchorID, memory: branchMemory))
    } else if let parent, alive.contains(parent) {
      setLeaf(parent)
    } else {
      setLeaf(messages.last?["id"]?.string)
    }
    return removed
  }
}
