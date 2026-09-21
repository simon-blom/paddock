import { describe, expect, it } from 'vitest'
import { getMarkdown, parseMarkdownToStructure, type CodeBlockNode, type ParsedNode } from 'stream-markdown-parser'
import { applyPatch, groupNodes, MarkdownParser, MAX_MARKDOWN_CHARS } from './parse'
import { languageFor, SyntaxRenderer } from './syntax'
import { parseMapBody } from '../mapblock'

const fixtures = [
  '# Title\n\n**bold** and _italic_ 東京 🐾\n\n> quote\n> - nested\n\n1. one\n2. two\n',
  '| A | B |\n| --- | --- |\n| 1 | **2** |\n\nTail',
  '```swift\nlet value = "<script>"\n```\n\n~~~~markdown\n```map\n1,2\n```\n~~~~\n',
  'Forward [link][later] and note[^n].\n\n[later]: https://example.com\n[^n]: A footnote\n',
  '$x^2$\n\n$$\n\\begin{pmatrix}1 & 2 \\\\ 3 & 4\\end{pmatrix}\n$$\n',
  '<script>alert(1)</script>\n\n[bad](javascript:alert(1))\n\n![image](data:text/html,unsafe)\n',
  '```mermaid\nflowchart LR\n A --> B\n```\n\nUnfinished **marker',
  '```diff typescript\n-const oldValue = 1\n+const newValue = 2\n```\n',
]
describe('worker-owned Markdown', () => {
  it('keeps maps inside the document grammar, without stealing longer example fences or late references', () => {
    const content = '[place][ref]\n\n~~~~markdown\n```map\n1,2,Example\n```\n~~~~\n\n```map\n59.3,18.1,Stockholm\n```\n\n[ref]: https://example.com\n'
    const result = new MarkdownParser().parse('map', { content, final: true, revision: 1, base: 0 })
    const fences = result.nodes.filter((node): node is CodeBlockNode => node.type === 'code_block')
    expect(fences.map(node => node.language)).toEqual(['markdown', 'map'])
    expect(fences[0].code).toContain('```map')
    expect(parseMapBody(fences[1].code)).toEqual({ lat: 59.3, lon: 18.1, label: 'Stockholm' })
    expect(JSON.stringify(result.nodes[0])).toContain('https://example.com')
    expect(parseMapBody('not coordinates')).toBeNull()
    expect(parseMapBody('{"lat": 99, "lon": 18}')).toBeNull()
  })
  it.each(fixtures)('streamed and rewritten AST equals one-shot parsing: %s', content => {
    const parser = new MarkdownParser()
    let nodes: ParsedNode[] = []; let revision = 0
    const inputs = Array.from({ length: Math.ceil(content.length / 7) }, (_, i) => content.slice(0, (i + 1) * 7))
    inputs.push(content.slice(0, 15), content, '')
    for (const text of inputs) {
      const next = revision + 1
      const patch = parser.parse('fixture', { content: text, final: false, revision: next, base: revision })
      nodes = applyPatch(nodes, revision, patch); revision = next
      expect(nodes).toEqual(parseMarkdownToStructure(text, getMarkdown('fixture'), { final: false, streamParse: false }))
    }
    const final = parser.parse('fixture', { content, final: true, revision: revision + 1, base: revision })
    expect(applyPatch(nodes, revision, final)).toEqual(parseMarkdownToStructure(content, getMarkdown('fixture'), { final: true, streamParse: false }))
    expect(parser.stats.sessions).toBe(0)
  })
  it.each([32, 64])('reuses exact leading objects and settled %i-node render groups', size => {
    const parser = new MarkdownParser()
    const prefix = 'Settled **paragraph**.\n\n'.repeat(size * 3)
    const a = parser.parse('a', { content: prefix + 'Tail', final: false, revision: 1, base: 0 })
    const groups = groupNodes(a.nodes, [], size)
    const b = parser.parse('a', { content: prefix + 'Tail extended', final: false, revision: 2, base: 1 })
    expect(b.start).toBeGreaterThanOrEqual(size * 3)
    const nodes = applyPatch(a.nodes, 1, b)
    const next = groupNodes(nodes, groups, size)
    expect(next[0]).toBe(groups[0]); expect(next[2]).toBe(groups[2]); expect(next[3]).not.toBe(groups[3])
    expect(next.flat()).toEqual(nodes)
    expect(b.serializedBytes).toBeLessThan(1024)
  })
  it('resynchronizes after a cancelled response, eviction, or worker restart', () => {
    const parser = new MarkdownParser()
    parser.parse('a', { content: 'one', final: false, revision: 1, base: 0 })
    parser.parse('a', { content: 'one two', final: false, revision: 2, base: 1 }) // caller discards
    expect(parser.parse('a', { content: 'one two three', final: false, revision: 3, base: 1 }).start).toBe(0)
    for (let i = 0; i < 32; i++) parser.parse(`other-${i}`, { content: 'text', final: false, revision: 1, base: 0 })
    expect(parser.stats.sessions).toBe(16)
    expect(parser.parse('a', { content: 'fresh', final: false, revision: 4, base: 3 }).start).toBe(0)
    expect(() => applyPatch([], 9, { revision: 10, base: 8, start: 1, nodes: [], total: 1, parseMs: 0, serializedBytes: 0 })).toThrow()
    expect(() => parser.parse('large', { content: 'x'.repeat(MAX_MARKDOWN_CHARS + 1), final: false, revision: 1, base: 0 })).toThrow('budget')
  })
  it('does not freeze a prefix when a late reference definition changes it', () => {
    const parser = new MarkdownParser()
    const first = parser.parse('ref', { content: '[label][target]\n\nSettled paragraph.', final: false, revision: 1, base: 0 })
    const content = '[label][target]\n\nSettled paragraph.\n\n[target]: https://example.com\n'
    const next = parser.parse('ref', { content, final: false, revision: 2, base: 1 })
    expect(next.start).toBe(0)
    expect(applyPatch(first.nodes, 1, next)).toEqual(parseMarkdownToStructure(content, getMarkdown('ref'), { final: false, streamParse: false }))
  })
})
describe('bounded syntax renderer', () => {
  it('supports aliases, escapes model HTML, caches output and keeps themes separate', async () => {
    expect(languageFor('TS')).toBe('typescript'); expect(languageFor('not-a-language')).toBe('text')
    const renderer = new SyntaxRenderer()
    const input = { code: 'const x = "<script>alert(1)</script>"', language: 'ts', dark: false }
    const html = await renderer.render(input)
    expect(html).toContain('color:'); expect(html).not.toContain('<script>')
    expect(await renderer.render(input)).toBe(html); expect(renderer.stats.entries).toBe(1)
    expect(await renderer.render({ ...input, dark: true })).not.toBe(html)
    await expect(renderer.render({ ...input, code: 'x'.repeat(9000) })).rejects.toThrow('without highlighting')
    for (let i = 0; i < 40; i++) await renderer.render({ ...input, code: `let n = ${i}` })
    expect(renderer.stats.entries).toBe(32); expect(renderer.stats.bytes).toBeLessThanOrEqual(4 * 1024 * 1024)
  })
})
