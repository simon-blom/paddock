import { test } from 'node:test'
import assert from 'node:assert/strict'
import { finalMemoryIdleSeconds, hasScreenLock, reopenCheckpoint } from './profile-guards.mjs'

test('final memory inspection preserves both original idle cutoffs', () => {
  assert.equal(finalMemoryIdleSeconds('scalegraphrepeat', 10), 0)
  assert.equal(finalMemoryIdleSeconds('scalegraph', 10), 10)
  assert.throws(() => finalMemoryIdleSeconds('scalegraph', 45), /original 10-second/)
  assert.throws(() => finalMemoryIdleSeconds('all', 10), /requires/)
})

test('reopen inspection stays inside the original cutoff and never repeats or accepts stale phases', () => {
  const phase = 'graph-cycle-5/idle-10s'
  const p = { phase, events: [{ phase, at: 1000 }] }
  const seen = new Set()
  assert.equal(reopenCheckpoint(p, seen, 10000), phase)
  for (const time of [0, 9999, 11000, 12000]) assert.equal(reopenCheckpoint(p, seen, time), null)
  assert.equal(reopenCheckpoint(undefined, seen, 10000), null)
  assert.equal(reopenCheckpoint({ ...p, phase: 'complete' }, seen, 10000), null)
  seen.add(phase); assert.equal(reopenCheckpoint(p, seen, 10000), null)
})

test('positive session lock prevents a visual profile without treating unrelated keys as lock', () => {
  assert.equal(hasScreenLock('"CGSSessionScreenIsLocked"=Yes'), true)
  assert.equal(hasScreenLock('"CGSSessionScreenIsLocked" = Yes, "other" = No'), true)
  assert.equal(hasScreenLock('"CGSSessionScreenIsLocked"=No'), false)
  assert.equal(hasScreenLock('"CGSSessionOnConsoleKey"=Yes'), false)
})
