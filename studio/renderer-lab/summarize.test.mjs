import { test } from 'node:test'
import assert from 'node:assert/strict'
import { summarize } from './summarize.mjs'

test('trigger diagnostics and explicit drop count survive summary export', () => {
  const graphTriggers = [{ kind: 'process', phase: 'x', start: 10, ms: 9, causes: ['graph:eachNodeAttributesUpdated'] }]
  const result = summarize({ profile: { events: [], metrics: [], graphTriggers, graphTriggersDropped: 3 } }, { processes: [] }, [])
  assert.deepEqual(result.graphTriggers, graphTriggers)
  assert.equal(result.graphTriggersDropped, 3)
})

test('post-cutoff native capture cannot improve any passive memory summary', () => {
  const report = { profile: { events: [{ phase: 'graph-cycle-5/idle-10s', at: 1000 }, { phase: 'complete', at: 11000 }], metrics: [] } }
  const run = { ended: 15000, memoryTracing: true, finalMemoryTracing: true, finalMemoryStarted: 11200,
    processes: [{ pid: 1, exitedWithLab: true }] }
  const sample = (at, mib) => ({ at, processes: [{ pid: 1, name: 'WebContent', footprint: mib * 1048576, peakFootprint: 600 * 1048576 }] })
  const r = summarize(report, run, [sample(10900, 528), sample(11100, 530), sample(13000, 100)])
  assert.equal(r.phases[0].lastMiB, 528)
  assert.equal(r.phases[1].lastMiB, 530)
  assert.ok(r.limitations.some(text => text.includes('No pre-cutoff native inspection')))
})

test('reopen allocation inspection is explicitly intrusive, never a latency bar', () => {
  const result = summarize({ profile: { events: [], metrics: [] } },
    { reopenTracing: true, memoryTracing: true, processes: [] }, [])
  assert.ok(result.limitations.some(text => text.includes('intrusive') && text.includes('not a latency bar')))
})

test('fresh final capture excludes inspection samples after the full external idle', () => {
  const report = { profile: { events: [{ phase: 'complete', at: 1000 }], metrics: [] } }
  const run = { plan: 'scalegraph', ended: 14000, memoryTracing: true, finalMemoryTracing: true,
    finalMemoryStarted: 11200, idleSeconds: 10, processes: [{ pid: 1, exitedWithLab: true }] }
  const sample = (at, mib) => ({ at, processes: [{ pid: 1, name: 'WebContent', footprint: mib * 1048576, peakFootprint: 600 * 1048576 }] })
  const result = summarize(report, run, [sample(5000, 400), sample(11000, 378), sample(12000, 100)])
  assert.equal(result.phases[0].lastMiB, 378)
  assert.equal(result.phases[0].sampledPeakMiB, 400)
})

test('layer inspection labels the entire run as diagnostic', () => {
  const report = { profile: { events: [], metrics: [] } }
  const result = summarize(report, { layerTracing: true, processes: [] }, [])
  assert.ok(result.limitations.some(text => text.includes('ENTIRE run is diagnostic')))
})

test('phase accounting uses concurrent footprint and excludes unverified processes', () => {
  const report = { profile: { events: [{ phase: 'load', at: 1000 }, { phase: 'close', at: 2000 }], metrics: [] } }
  const run = { ended: 3000, processes: [{ pid: 1, exitedWithLab: true }, { pid: 2, exitedWithLab: true }, { pid: 3, exitedWithLab: false }] }
  const p = (pid, mib) => ({ pid, name: `process${pid}`, footprint: mib * 1048576, peakFootprint: mib * 1048576 })
  const samples = [{ at: 1500, processes: [p(1, 10), p(2, 1), p(3, 1000)] }, { at: 1800, processes: [p(1, 1), p(2, 10)] }, { at: 2200, processes: [p(1, 2)] }]
  const result = summarize(report, run, samples)
  assert.equal(result.phases[0].sampledPeakMiB, 11) // not 20, not 1011
  assert.equal(result.phases[1].lastMiB, 2)
})

test('heap-inspector processes and post-GC samples cannot improve the unforced summary', () => {
  const report = { profile: { events: [{ phase: 'complete', at: 1000 }], metrics: [] } }
  const run = { ended: 4000, memoryTracing: true, heapSnapshot: true, heapCapture: { started: 2500 },
    processes: [{ pid: 1, exitedWithLab: true }, { pid: 2, exitedWithLab: true }] }
  const p = (pid, mib) => ({ pid, name: `process${pid}`, footprint: mib * 1048576, peakFootprint: mib * 1048576 })
  const samples = [{ at: 2000, processes: [p(1, 200)] }, { at: 3000, processes: [p(1, 20), p(2, 1000)] }]
  const result = summarize(report, run, samples)
  assert.equal(result.phases[0].lastMiB, 200)
  assert.equal(result.processPeaks.length, 1)
  assert.ok(result.limitations.some(text => text.includes('forces GC')))
})
