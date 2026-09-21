import type Graph from 'graphology'
import type Sigma from 'sigma'

export interface HoverState { node: string | null; nodes: Set<string>; edges: Set<string> }
const empty = (): HoverState => ({ node: null, nodes: new Set(), edges: new Set() })
const changed = (before: Set<string>, after: Set<string>) =>
  [...before].filter(key => !after.has(key)).concat([...after].filter(key => !before.has(key)))

/** Hover changes colors, label visibility and forceLabel, never geometry or
 * program type. Coalesce leave/enter before calling Sigma: scheduleRefresh()
 * only defers its render; it still rebuilds the data cache on every call.
 *
 * GraphCanvas preserves string/null label membership when fading labels to
 * empty strings. Sigma 3 indexes even empty strings, so hover never changes
 * label-grid membership, positions or ordering. Both entry and exit can use
 * partial repaints. Graph mutations conservatively take the ordinary full
 * path until processed; callers must preserve this reducer invariant.
 * No private Sigma fields, monkey-patched scheduling or suppressed picking. */
export function createHoverRefresh(renderer: Sigma, graph: Graph) {
  let applied = empty(), desired = applied
  let pending = false
  let dirty = false, disposed = false
  const events = ['nodeAdded', 'nodeDropped', 'edgeAdded', 'edgeDropped', 'cleared', 'edgesCleared',
    'nodeAttributesUpdated', 'edgeAttributesUpdated', 'eachNodeAttributesUpdated', 'eachEdgeAttributesUpdated'] as const
  const invalidate = () => { dirty = true }
  const processed = () => { dirty = false }
  for (const event of events) graph.on(event, invalidate)
  renderer.on('afterProcess', processed)
  // Sigma invalidates its indexes on window resize, before emitting its own
  // resize event from render(). Observe the same source, not the later event.
  if (typeof window !== 'undefined') window.addEventListener('resize', invalidate)
  const dispose = () => {
    if (disposed) return
    disposed = true
    pending = false; applied = empty(); desired = applied
    for (const event of events) graph.removeListener(event, invalidate)
    renderer.removeListener('afterProcess', processed)
    renderer.removeListener('beforeRender', flush)
    renderer.removeListener('kill', dispose)
    if (typeof window !== 'undefined') window.removeEventListener('resize', invalidate)
  }
  const flush = () => {
    if (!pending || disposed) return
    pending = false
    // Fold the appearance invalidation into Sigma's next render, before its
    // pending geometry/index pass. A separate rAF could run after that pass
    // and cause a second cache rebuild + draw in the same frame. Scheduling
    // from this public hook is non-recursive; the in-progress render consumes
    // the invalidation. No call to Sigma's private render/process methods.
    const local = applied.node && desired.node
    const nodes = local
      ? [...new Set([...changed(applied.nodes, desired.nodes), applied.node, desired.node])].filter((key): key is string => key !== null && graph.hasNode(key))
      : graph.nodes()
    const edges = local ? changed(applied.edges, desired.edges).filter(key => graph.hasEdge(key)) : graph.edges()
    // Graphology already updated the caches of mutated entities. Only the
    // hover delta needs reducing again. A dirty index must not be written via
    // the repaint path (new/type-changed programs may have no indexes yet).
    renderer.refresh({ partialGraph: { nodes, edges }, skipIndexation: !dirty, schedule: true })
    applied = desired
  }
  renderer.on('beforeRender', flush)
  renderer.on('kill', dispose)
  return {
    update(state: HoverState) {
      if (disposed) return
      desired = state
      pending = true
      renderer.scheduleRender()
    },
    /** Call before changing Sigma settings, whose refresh also clears indexes. */
    invalidate,
    dispose,
  }
}
