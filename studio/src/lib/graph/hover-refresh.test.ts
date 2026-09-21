import { describe, expect, it, vi } from 'vitest'
import Graph from 'graphology'
import type Sigma from 'sigma'
import { createHoverRefresh, type HoverState } from './hover-refresh'

function harness() {
  const graph = new Graph()
  for (const key of ['a', 'b', 'c', 'd']) graph.addNode(key)
  graph.addEdgeWithKey('ab', 'a', 'b'); graph.addEdgeWithKey('bc', 'b', 'c'); graph.addEdgeWithKey('cd', 'c', 'd')
  const listeners = new Map<string, () => void>(), frames = new Map<number, () => void>()
  const renderer = { refresh: vi.fn(), on: (key: string, fn: () => void) => listeners.set(key, fn),
    scheduleRender: () => frames.set(1, () => listeners.get('beforeRender')?.()),
    removeListener: (key: string) => listeners.delete(key) } as unknown as Sigma
  const refresh = createHoverRefresh(renderer, graph)
  const state = (node: string | null): HoverState => ({ node, nodes: new Set(node ? [node, ...graph.neighbors(node)] : []), edges: new Set(node ? graph.edges(node) : []) })
  return { graph, renderer, refresh, listeners, frames, state, flush: () => { const pending = [...frames.values()]; frames.clear(); for (const fn of pending) fn() } }
}
describe('coalesced graph hover refresh', () => {
  it('defers cache work as well as drawing, fading everything only on initial enter', () => {
    const h = harness(); h.refresh.update(h.state('a'))
    expect(h.renderer.refresh).not.toHaveBeenCalled()
    h.flush()
    expect(h.renderer.refresh).toHaveBeenCalledExactlyOnceWith({ partialGraph: { nodes: ['a', 'b', 'c', 'd'], edges: ['ab', 'bc', 'cd'] }, skipIndexation: true, schedule: true })
    h.refresh.dispose()
  })
  it('coalesces leave/enter and repaints only changed neighborhoods plus hovered label colors', () => {
    const h = harness(); h.refresh.update(h.state('a')); h.flush()
    vi.mocked(h.renderer.refresh).mockClear()
    h.refresh.update(h.state(null)); h.refresh.update(h.state('b'))
    expect(h.frames.size).toBe(1); h.flush()
    expect(h.renderer.refresh).toHaveBeenCalledExactlyOnceWith({ partialGraph: { nodes: ['c', 'a', 'b'], edges: ['bc'] }, skipIndexation: true, schedule: true })
    h.refresh.dispose()
  })
  it('restores all appearance data without rebuilding the unchanged label grid on exit', () => {
    const h = harness(); h.refresh.update(h.state('a')); h.flush()
    vi.mocked(h.renderer.refresh).mockClear()
    h.refresh.update(h.state(null)); h.flush()
    expect(h.renderer.refresh).toHaveBeenCalledExactlyOnceWith({ partialGraph: { nodes: ['a', 'b', 'c', 'd'], edges: ['ab', 'bc', 'cd'] }, skipIndexation: true, schedule: true })
    h.refresh.dispose()
  })
  it('never reuses program indexes across pending graph mutation', () => {
    const h = harness(); h.refresh.update(h.state('a')); h.flush()
    h.graph.addNode('e'); h.refresh.update(h.state('b')); h.flush()
    expect(h.renderer.refresh).toHaveBeenLastCalledWith(expect.objectContaining({ skipIndexation: false, schedule: true }))
    h.listeners.get('afterProcess')!()
    h.refresh.update(h.state('c')); h.flush()
    expect(h.renderer.refresh).toHaveBeenLastCalledWith(expect.objectContaining({ skipIndexation: true }))
    h.graph.setNodeAttribute('a', 'type', 'pulse'); h.refresh.update(h.state('a')); h.flush()
    expect(h.renderer.refresh).toHaveBeenLastCalledWith(expect.objectContaining({ skipIndexation: false, schedule: true }))
    h.refresh.dispose()
  })
  it('kill cancels pending work and removes graph listeners and retained neighborhoods', () => {
    const h = harness(); h.refresh.update(h.state('a'))
    expect(h.graph.listenerCount('nodeAdded')).toBe(1)
    h.listeners.get('kill')!(); h.flush()
    expect(h.renderer.refresh).not.toHaveBeenCalled(); expect(h.frames.size).toBe(0)
    expect(h.graph.listenerCount('nodeAdded')).toBe(0); expect(h.listeners.size).toBe(0)
    h.refresh.update(h.state('b')); expect(h.frames.size).toBe(0)
    h.refresh.dispose()
  })
  it('settings invalidation folds the hover delta into one ordinary index pass', () => {
    const h = harness(); h.refresh.update(h.state('a')); h.flush()
    vi.mocked(h.renderer.refresh).mockClear()
    h.refresh.invalidate(); h.refresh.update(h.state('b')); h.flush()
    expect(h.renderer.refresh).toHaveBeenCalledExactlyOnceWith({ partialGraph: { nodes: ['c', 'a', 'b'], edges: ['bc'] }, skipIndexation: false, schedule: true })
    h.refresh.dispose()
  })
})
