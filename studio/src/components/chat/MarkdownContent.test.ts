import { afterEach, describe, expect, it, vi } from 'vitest'
import { createRenderer, defineComponent, h, nextTick, ref, ssrContextKey, type Component } from 'vue'
import MarkdownContent from './MarkdownContent.vue'
import { markdownParser } from '@/lib/markdown/runtime'
import { markdownMounts } from '@/lib/markdown/mount-scheduler'
import { MAX_MARKDOWN_CHARS } from '@/lib/markdown/parse'

vi.mock('@/lib/markstream', () => ({ initMarkstream: () => {} }))
// Exercise the production component's watcher/lifecycle with real WorkerRpc
// and scheduler. Native lab tests separately cover Markstream's actual DOM.
vi.mock('markstream-vue', () => ({ MarkdownRender: defineComponent({
  props: ['nodes'], setup: () => () => h('p'),
}) }))

class Port {
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror = null
  onmessageerror = null
  terminated = false
  sent: { id: number; owner: string; input: { revision: number; base: number } }[] = []
  constructor() { ports.push(this) }
  postMessage(message: typeof this.sent[number]) { this.sent.push(message) }
  terminate() { this.terminated = true }
  reply() {
    const { id, input } = this.sent[this.sent.length - 1]
    this.onmessage?.({ data: { id, value: { revision: input.revision, base: 0, start: 0,
      total: 1, nodes: [{ type: 'paragraph', children: [{ type: 'text', content: 'reply' }] }],
      parseMs: 0, serializedBytes: 100 } } } as MessageEvent)
  }
}
let ports: Port[] = []
interface HostNode { children: HostNode[]; parent: HostNode | null }
const node = (): HostNode => ({ children: [], parent: null })
const remove = (child: HostNode) => {
  const siblings = child.parent?.children
  if (siblings) siblings.splice(siblings.indexOf(child), 1)
  child.parent = null
}
const renderer = createRenderer<HostNode, HostNode>({
  createElement: node, createText: node, createComment: node,
  insert(child, parent, anchor) {
    remove(child); child.parent = parent
    parent.children.splice(anchor ? parent.children.indexOf(anchor) : parent.children.length, 0, child)
  },
  remove, parentNode: child => child.parent,
  nextSibling: child => child.parent?.children[child.parent.children.indexOf(child) + 1] ?? null,
  patchProp: () => {}, setText: () => {}, setElementText: () => {},
})
const unmounts: (() => void)[] = []
// Vitest's Node transform emits an SSR template. Mount the real setup and
// lifecycle on a tiny host, not that template; native tests own DOM assertions.
const Subject = { ...MarkdownContent, render: () => h('div') } as Component
function mount(contents: string[]) {
  ports = []; vi.useFakeTimers(); vi.stubGlobal('Worker', Port)
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => setTimeout(() => callback(performance.now()), 16))
  vi.stubGlobal('cancelAnimationFrame', clearTimeout)
  const values = ref(contents)
  const app = renderer.createApp(() => h('div', values.value.map((content, key) => h(Subject, { key, content, streaming: true }))))
  app.provide(ssrContextKey, { modules: new Set() })
  app.mount(node()); unmounts.push(() => app.unmount())
  return values
}
afterEach(async () => {
  for (const unmount of unmounts.splice(0)) unmount()
  await nextTick(); await Promise.resolve()
  expect(markdownParser.stats).toMatchObject({ workers: 0, owners: 0, active: 0, queued: 0, waiting: 0, bytes: 0 })
  expect(markdownMounts.stats.owners).toBe(0)
  vi.useRealTimers(); vi.unstubAllGlobals()
})

describe('Markdown parser consumer lifetime', () => {
  it('does not acquire a lease for mounted empty or oversized plain views', () => {
    mount(['', '', 'x'.repeat(MAX_MARKDOWN_CHARS + 1)])
    expect(markdownParser.stats.owners).toBe(0)
    expect(ports).toHaveLength(0)
  })
  it('releases the last populated consumer even with empty siblings still mounted', async () => {
    const values = mount(['', '**one**', ''])
    expect(markdownParser.stats.owners).toBe(1); ports[0].reply()
    await nextTick(); await vi.advanceTimersByTimeAsync(32)
    values.value[1] = ''; await nextTick()
    expect(ports[0].terminated).toBe(true)
    expect(markdownParser.stats.owners).toBe(0)
    values.value[2] = '**reopened**'; await nextTick()
    expect(ports).toHaveLength(2); expect(markdownParser.stats.owners).toBe(1)
    expect(ports[1].sent[0].input.base).toBe(0)
    expect(ports[1].sent[0].owner).not.toBe(ports[0].sent[0].owner)
    ports[1].reply(); await nextTick(); await vi.advanceTimersByTimeAsync(32)
  })
  it('keeps another populated consumer and its physical job alive on clear', async () => {
    const values = mount(['**one**', '**two**', ''])
    const owner = ports[0].sent[0].owner
    expect(markdownParser.stats.owners).toBe(2)
    values.value[0] = ''; await nextTick()
    expect(ports[0].terminated).toBe(false)
    expect(markdownParser.stats).toMatchObject({ owners: 1, active: 1, queued: 1 })
    ports[0].reply() // cancelled caller, but its physical slot was not reused
    expect(ports[0].sent[1].owner).not.toBe(owner)
    ports[0].reply(); await nextTick(); await vi.advanceTimersByTimeAsync(32)
    expect(ports).toHaveLength(1)
  })
  it('rejects an old reply after clear/reopen without touching the new owner', async () => {
    const values = mount(['**old**'])
    const reply = ports[0].onmessage!
    const old = ports[0].sent[0]
    values.value[0] = ''; await nextTick()
    values.value[0] = '**new**'; await nextTick()
    reply({ data: { id: old.id, value: { revision: old.input.revision } } } as MessageEvent)
    expect(markdownParser.stats).toMatchObject({ workers: 1, owners: 1, active: 1 })
    ports[1].reply(); await nextTick(); await vi.advanceTimersByTimeAsync(32)
    expect(markdownMounts.stats.owners).toBe(0)
  })
  it('drops the lease and queued mounting when content becomes plain-only', async () => {
    const values = mount(['**one**']); ports[0].reply(); await nextTick()
    values.value[0] = 'x'.repeat(MAX_MARKDOWN_CHARS + 1); await nextTick()
    expect(ports[0].terminated).toBe(true)
    expect(markdownParser.stats.owners).toBe(0)
    expect(markdownMounts.stats.owners).toBe(0)
  })
  it('preserves the 15-second idle policy for a remaining populated view', async () => {
    mount(['**one**', '']); ports[0].reply(); await nextTick()
    await vi.advanceTimersByTimeAsync(14999); expect(ports[0].terminated).toBe(false)
    await vi.advanceTimersByTimeAsync(1); expect(ports[0].terminated).toBe(true)
    expect(markdownParser.stats.owners).toBe(1)
  })
})
