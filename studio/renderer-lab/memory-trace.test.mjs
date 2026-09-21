import { test } from 'node:test'
import assert from 'node:assert/strict'
import { MemoryTrace, parseFootprint } from './memory-trace.mjs'

test('footprint parser separates dirty footprint from reclaimable and clean storage', () => {
  const summary = parseFootprint(`WebContent [12]: Footprint: 300 B (16384 bytes per page)
  200 B 0 B 800 B 12 WebKit malloc
  100 B 40 B 50 B 2 Owned physical footprint (unmapped) (graphics)
  300 B 40 B 850 B 14 TOTAL`)
  assert.equal(summary.footprintBytes, 300)
  assert.equal(summary.categories.length, 2)
  assert.equal(summary.categories[0].reclaimableBytes, 800)
  assert.equal(summary.categories[1].name, 'Owned physical footprint (unmapped) (graphics)')
  assert.throws(() => parseFootprint('permission denied'), /Unrecognized/)
})
test('native tracing refuses pre-existing and ambiguous WebContent targets', async () => {
  const p = pid => ({ pid, path: '/System/com.apple.WebKit.WebContent' })
  const trace = new MemoryTrace('/unused', [p(1)], () => [p(1)])
  await trace.capture('baseline')
  assert.match(trace.rows[0].error, /found 0/)
  trace.processList = () => [p(1), p(2), p(3)]
  await trace.capture('post-0s')
  assert.match(trace.rows[1].error, /found 2/)
  await assert.rejects(() => trace.capture('../elsewhere'), /Invalid/)
})
