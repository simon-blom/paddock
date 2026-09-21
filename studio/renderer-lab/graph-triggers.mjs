/** Lab-only causal tags at the actual refresh/process boundary. No stacks,
 * graph keys or payloads are retained. Wrappers preserve calls and scheduling. */
export function traceGraphTriggers(prototype, graph, record, now = () => performance.now(), discovered = () => {}) {
  const restore = []
  const pending = new WeakMap()
  const processing = new WeakMap()
  const seen = new WeakSet()
  let cause = 'unclassified'
  const scoped = (name, fn) => {
    const previous = cause; cause = name
    try { return fn() } finally { cause = previous }
  }
  const wrap = (object, key, fn) => {
    const own = Object.getOwnPropertyDescriptor(object, key)
    const original = object[key]
    object[key] = function (...args) { return fn(this, original, args) }
    restore.push(() => { if (own) Object.defineProperty(object, key, own); else delete object[key] })
  }
  wrap(graph, 'emit', (self, original, args) => scoped(`graph:${args[0]}`, () => original.apply(self, args)))
  for (const key of ['setSetting', 'updateSetting']) wrap(prototype, key, (self, original, args) =>
    scoped(`setting:${args[0]}`, () => original.apply(self, args)))
  wrap(prototype, 'refresh', (self, original, args) => {
    if (self.getGraph() !== graph) return original.apply(self, args)
    if (!seen.has(self)) { seen.add(self); discovered(self) }
    const options = args[0]
    const full = !options?.partialGraph
    const reindex = full || !options?.skipIndexation
    const start = now()
    if (reindex) {
      let causes = pending.get(self)
      if (!causes) pending.set(self, causes = new Set())
      causes.add(cause)
    }
    const trigger = cause
    try { return original.apply(self, args) }
    finally { record({ kind: 'refresh', start, ms: now() - start, causes: [trigger], full, reindex,
      scheduled: !!options?.schedule, nodes: options?.partialGraph?.nodes?.length ?? null,
      edges: options?.partialGraph?.edges?.length ?? null }) }
  })
  const interactions = new Set(['enterNode', 'leaveNode', 'downNode', 'moveBody', 'upNode', 'upStage'])
  wrap(prototype, 'emit', (self, original, args) => {
    if (self.getGraph() !== graph) return original.apply(self, args)
    if (interactions.has(args[0])) record({ kind: 'interaction', start: now(), ms: 0, causes: [`interaction:${args[0]}`] })
    if (args[0] === 'beforeProcess') {
      processing.set(self, { start: now(), causes: [...(pending.get(self) ?? ['unclassified'])] })
      pending.delete(self)
    }
    try {
      return interactions.has(args[0])
        ? scoped(`interaction:${args[0]}`, () => original.apply(self, args))
        : original.apply(self, args)
    } finally {
      if (args[0] === 'afterProcess') {
        const row = processing.get(self)
        if (row) { record({ kind: 'process', ...row, ms: now() - row.start }); processing.delete(self) }
      }
    }
  })
  return () => { for (const undo of restore.reverse()) undo() }
}
