import { getMarkdown, parseMarkdownToStructure, type ParsedNode } from 'stream-markdown-parser'

export const MAX_MARKDOWN_CHARS = 1024 * 1024
const MAX_AST_BYTES = 8 * 1024 * 1024
const MAX_CACHE_BYTES = 16 * 1024 * 1024
export interface ParseInput { content: string; final: boolean; revision: number; base: number }
export interface ParseOutput {
  revision: number; base: number; start: number; nodes: ParsedNode[]; total: number
  parseMs: number; serializedBytes: number
}
interface Session {
  md: ReturnType<typeof getMarkdown>; signatures: string[]; revision: number; bytes: number
}

// Full-document parsing preserves global reference/footnote semantics. Only
// exact-equal leading AST nodes are reused; no truncated hashes, blank-line
// heuristics, or assumption that an append cannot change earlier content.
export class MarkdownParser {
  private sessions = new Map<string, Session>()
  get stats() { return { sessions: this.sessions.size, bytes: [...this.sessions.values()].reduce((n, s) => n + s.bytes, 0) } }
  parse(owner: string, input: ParseInput): ParseOutput {
    if (input.content.length > MAX_MARKDOWN_CHARS) throw new Error('Markdown exceeds rich-rendering budget; showing complete plain text')
    const before = this.sessions.get(owner)
    this.sessions.delete(owner)
    // Final passes are deliberately one-shot. Streaming cache bugs or a final
    // delimiter can never leave the completed answer with provisional syntax.
    const started = performance.now()
    const md = !input.final && before ? before.md : getMarkdown(owner)
    const nodes = parseMarkdownToStructure(input.content, md, {
      final: input.final, streamParse: !input.final, reuseStableTopLevelNodes: !input.final,
    })
    const signatures = nodes.map(node => JSON.stringify(node))
    const astBytes = signatures.reduce((sum, value) => sum + value.length * 2, 0)
    if (astBytes > MAX_AST_BYTES || nodes.length > 16384) throw new Error('Markdown AST exceeds rich-rendering budget; showing complete plain text')
    let start = 0
    if (before?.revision === input.base) {
      while (start < Math.min(signatures.length, before.signatures.length) && signatures[start] === before.signatures[start]) start++
    }
    const output = { revision: input.revision, base: start ? input.base : 0, start,
      nodes: nodes.slice(start), total: nodes.length, parseMs: performance.now() - started,
      serializedBytes: signatures.slice(start).reduce((sum, s) => sum + s.length * 2, 0) }
    // Estimated retained source + AST accounting, not a bound on the JS heap.
    const bytes = input.content.length * 2 + astBytes
    if (!input.final && bytes <= MAX_CACHE_BYTES) {
      this.sessions.set(owner, { md, signatures, revision: input.revision, bytes })
      while (this.sessions.size > 16 || this.stats.bytes > MAX_CACHE_BYTES) this.sessions.delete(this.sessions.keys().next().value!)
    }
    return output
  }
}

export function applyPatch(previous: ParsedNode[], revision: number, patch: ParseOutput): ParsedNode[] {
  if (patch.start < 0 || patch.start > previous.length || patch.total !== patch.start + patch.nodes.length
    || (patch.start > 0 && patch.base !== revision)) throw new Error('Markdown patch revision mismatch')
  return [...previous.slice(0, patch.start), ...patch.nodes]
}

// Immutable groups limit component updates to changed groups (the reference
// scan itself is still O(N)). This is not transcript virtualization.
export function groupNodes(nodes: ParsedNode[], previous: ParsedNode[][], size = 32): ParsedNode[][] {
  const groups: ParsedNode[][] = []
  for (let offset = 0; offset < nodes.length; offset += size) {
    const old = previous[groups.length]
    const count = Math.min(size, nodes.length - offset)
    groups.push(old?.length === count && old.every((node, i) => node === nodes[offset + i]) ? old : nodes.slice(offset, offset + size))
  }
  return groups
}
