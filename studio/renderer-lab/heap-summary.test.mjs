import { test } from 'node:test'
import assert from 'node:assert/strict'
import { summarizeHeap } from './heap-summary.mjs'

const fixture = () => ({ version: 2, type: 'Inspector',
  nodes: [0, 0, 0, 0, 7, 16, 1, 0, 11, 24, 2, 0, 31, 32, 3, 4, 55, 32, 3, 4],
  nodeClassNames: ['<root>', 'Window', 'Map', 'HTMLSpanElement'],
  edges: [0, 7, 0, 0, 7, 11, 1, 0, 11, 31, 2, 0, 31, 11, 0, 0],
  edgeTypes: ['Internal', 'Property', 'Index', 'Variable'], edgeNames: ['cache'],
})
test('heap analysis follows a shortest root path through cycles and preserves counts', () => {
  const report = summarizeHeap(fixture())
  assert.equal(report.nodeCount, 5)
  assert.equal(report.selfBytes, 104)
  assert.equal(report.selected[0].count, 2)
  assert.deepEqual(report.selected[0].path.map(n => n.id), [0, 7, 11, 31])
  assert.equal(report.selected[0].path[2].name, 'cache')
  assert.equal(report.selected[0].path[3].via, 'Index')
})
test('heap analysis rejects GC-debug snapshots with a different node stride', () => {
  assert.throws(() => summarizeHeap({ ...fixture(), type: 'GCDebugging' }), /Unsupported/)
})
