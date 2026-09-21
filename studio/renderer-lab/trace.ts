// Development-only, bounded main-thread attribution. Timings are inclusive;
// WebGL calls measure CPU submission/backpressure, not GPU execution time.
import type Graph from 'graphology'
import Sigma from 'sigma'
import { traceGraphTriggers } from './graph-triggers.mjs'
import { metric, profile } from './report'

export class StageTrace {
  renderer?: WeakRef<Sigma>
  private rows = new Map<string, { name: string; phase: string; count: number; total: number; max: number; samples: number[]; bytes: number }>()
  private restore: (() => void)[] = []
  readonly record = (name: string, start: number, bytes = 0) => {
    const ms = performance.now() - start
    const key = `${profile.phase}/${name}`
    let row = this.rows.get(key)
    if (!row) this.rows.set(key, row = { name, phase: profile.phase, count: 0, total: 0, max: 0, samples: [], bytes: 0 })
    row.count++; row.total += ms; row.max = Math.max(row.max, ms); row.bytes += bytes
    if (row.samples.length < 8192) row.samples.push(ms)
    if (ms >= 8 && profile.spans.length < 512) profile.spans.push({ name, start, ms, phase: profile.phase })
  }
  wrap(object: object, key: string, name: string, byteSize?: (args: unknown[]) => number) {
    const target = object as Record<string, (...args: any[]) => any>
    const original = target[key]
    if (typeof original !== 'function') throw new Error(`Trace method missing: ${key}`)
    const own = Object.getOwnPropertyDescriptor(object, key)
    const record = this.record
    target[key] = function (...args) {
      const start = performance.now()
      try { return original.apply(this, args) }
      finally { record(name, start, byteSize?.(args) ?? 0) }
    }
    this.restore.push(() => { if (own) Object.defineProperty(object, key, own); else delete target[key] })
  }
  graph(graph: Graph) {
    this.wrap(graph, 'updateEachNodeAttributes', 'graph-position-commit')
    this.restore.push(traceGraphTriggers(Sigma.prototype, graph, (row: object) => {
      if (profile.graphTriggers.length < 768) profile.graphTriggers.push({ ...row, phase: profile.phase })
      else profile.graphTriggersDropped++
    }, undefined, (renderer: Sigma) => { this.renderer = new WeakRef(renderer) }))
  }
  webgl() {
    const seen = new WeakSet<object>()
    const original = HTMLCanvasElement.prototype.getContext
    const trace = this
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, ...args: any[]) {
      const context = (original as Function).apply(this, args)
      if (context && ['webgl', 'webgl2', 'experimental-webgl'].includes(args[0]) && !seen.has(context)) {
        seen.add(context)
        for (const name of ['bufferData', 'bufferSubData']) trace.wrap(context, name, `webgl-${name}-cpu`, a => {
          const data = a[name === 'bufferData' ? 1 : 2]
          return typeof data === 'number' ? data : ArrayBuffer.isView(data) ? data.byteLength : data instanceof ArrayBuffer ? data.byteLength : 0
        })
        for (const name of ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced', 'texImage2D']) {
          if (typeof context[name] === 'function') trace.wrap(context, name, `webgl-${name}-cpu`)
        }
        metric('gpu-timer-query-available', context.getExtension('EXT_disjoint_timer_query_webgl2') || context.getExtension('EXT_disjoint_timer_query') ? 1 : 0, 'boolean')
      }
      return context
    } as typeof original
    this.restore.push(() => { HTMLCanvasElement.prototype.getContext = original })
  }
  finish() {
    this.renderer = undefined
    for (const undo of this.restore.reverse()) undo()
    this.restore = []
    for (const row of this.rows.values()) {
      const { name, phase } = row
      const sorted = row.samples.sort((a, b) => a - b)
      metric(`${name}-count`, row.count, 'count', phase); metric(`${name}-total`, row.total, 'ms', phase)
      metric(`${name}-max`, row.max, 'ms', phase); metric(`${name}-p99`, sorted[Math.max(0, Math.ceil(sorted.length * .99) - 1)] ?? 0, 'ms', phase)
      if (row.bytes) metric(`${name}-bytes`, row.bytes, 'bytes', phase)
    }
  }
}
