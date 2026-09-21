import { test } from 'node:test'
import assert from 'node:assert/strict'
import { traceGraphTriggers } from './graph-triggers.mjs'

function harness() {
  let time = 0
  const graph = { emit() { renderer.refresh({ partialGraph: { nodes: ['a'] }, schedule: true }); return true } }
  class Renderer {
    getGraph() { return graph }
    emit(event) { if (event === 'enterNode') this.refresh(); return true }
    setSetting() { return this.refresh({ schedule: true }) }
    updateSetting() { throw new Error('setting failed') }
    refresh(options) { if (!options?.schedule) this.process(); return this }
    process() { this.emit('beforeProcess'); time += 10; this.emit('afterProcess') }
  }
  const renderer = new Renderer(), rows = []
  const restore = traceGraphTriggers(Renderer.prototype, graph, row => rows.push(row), () => time)
  return { renderer, graph, rows, restore, Renderer }
}
test('preserves returns and attributes deferred processing to accumulated causes', () => {
  const h = harness()
  assert.equal(h.graph.emit('eachNodeAttributesUpdated'), true)
  assert.equal(h.renderer.setSetting('renderLabels', true), h.renderer)
  h.renderer.process()
  assert.deepEqual(h.rows.at(-1), { kind: 'process', start: 0, ms: 10, causes: ['graph:eachNodeAttributesUpdated', 'setting:renderLabels'] })
  h.restore()
})
test('attributes synchronous hover processing and keeps skipIndexation semantics honest', () => {
  const h = harness()
  assert.equal(h.renderer.emit('enterNode'), true)
  assert.equal(h.rows[0].kind, 'interaction')
  assert.deepEqual(h.rows[1].causes, ['interaction:enterNode'])
  h.renderer.refresh({ skipIndexation: true, schedule: true })
  assert.equal(h.rows.at(-1).reindex, true) // full refresh still processes in Sigma 3
  h.renderer.refresh({ partialGraph: { nodes: ['a'] }, skipIndexation: true, schedule: true })
  assert.equal(h.rows.at(-1).reindex, false)
  h.restore()
})
test('restores exception scope, methods and inherited descriptors without retaining graph data', () => {
  const h = harness()
  assert.throws(() => h.renderer.updateSetting(), /setting failed/)
  h.renderer.refresh()
  assert.deepEqual(h.rows[0].causes, ['unclassified'])
  h.restore()
  const count = h.rows.length
  h.renderer.refresh(); h.graph.emit('eachNodeAttributesUpdated')
  assert.equal(h.rows.length, count)
  assert.equal(Object.hasOwn(h.Renderer.prototype, 'refresh'), true)
  assert.ok(!JSON.stringify(h.rows).includes('"a"'))
})
