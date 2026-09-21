import { describe, expect, it } from 'vitest'
import { MountScheduler } from './mount-scheduler'

function harness() {
  let now = 0; let id = 0
  const frames = new Map<number, FrameRequestCallback>()
  const scheduler = new MountScheduler({ now: () => now,
    frame: callback => { frames.set(++id, callback); return id }, cancel: key => { frames.delete(key) },
  }, 4)
  const run = async () => {
    const callbacks = [...frames.values()]; frames.clear()
    for (const callback of callbacks) callback(now)
    for (let i = 0; i < 100; i++) await Promise.resolve()
  }
  return { scheduler, frames, run, cost: (ms: number) => { now += ms } }
}
describe('shared Markdown frame budget', () => {
  it('charges asynchronous DOM flushes and rotates owners across one shared frame', async () => {
    const h = harness(); const order: string[] = []; const counts = new Map<string, number>()
    for (const owner of ['a', 'b', 'c', 'd']) h.scheduler.schedule(owner, {
      step: async () => { await Promise.resolve(); h.cost(3); order.push(owner); counts.set(owner, (counts.get(owner) ?? 0) + 1); return counts.get(owner)! < 2 },
      error: error => { throw error },
    })
    expect(h.frames.size).toBe(1)
    await h.run(); expect(order).toEqual(['a', 'b']); expect(h.scheduler.stats.maxFrameMs).toBe(6)
    await h.run(); expect(order).toEqual(['a', 'b', 'c', 'd'])
    await h.run(); await h.run()
    expect(order).toEqual(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'])
    expect(h.scheduler.stats.owners).toBe(0); expect(h.frames.size).toBe(0)
  })
  it('replacement/cancellation during a flush cannot resurrect obsolete work', async () => {
    const h = harness(); const calls: string[] = []
    h.scheduler.schedule('a', { step: async () => {
      calls.push('old'); h.cost(4)
      h.scheduler.schedule('a', { step: async () => { calls.push('new'); return false }, error: () => {} })
      return true
    }, error: () => {} })
    await h.run(); await h.run(); expect(calls).toEqual(['old', 'new'])
    h.scheduler.schedule('a', { step: async () => { h.scheduler.cancel('a'); return true }, error: () => {} })
    await h.run(); expect(h.frames.size).toBe(0); expect(h.scheduler.stats.owners).toBe(0)
  })
  it('coalesces repeated admission, stops idle frames, and recovers after an error', async () => {
    const h = harness(); let errors = 0; let calls = 0
    for (let i = 0; i < 500; i++) h.scheduler.schedule('a', { step: async () => { calls++; return false }, error: () => {} })
    expect(h.scheduler.stats.owners).toBe(1)
    h.scheduler.cancel('a'); expect(h.frames.size).toBe(0)
    h.scheduler.schedule('bad', { step: async () => { throw new Error('test') }, error: () => { errors++ } })
    h.scheduler.schedule('good', { step: async () => { calls++; return false }, error: () => {} })
    await h.run(); expect(errors).toBe(1); expect(calls).toBe(1); expect(h.frames.size).toBe(0)
  })
  it('eager warm work debits the same budget until the next frame', async () => {
    const h = harness(); const calls: string[] = []
    const job = (name: string) => ({ step: async () => { calls.push(name); h.cost(3); return false }, error: () => {} })
    h.scheduler.schedule('a', job('a'), true)
    for (let i = 0; i < 10; i++) await Promise.resolve()
    expect(calls).toEqual(['a'])
    h.scheduler.schedule('b', job('b'), true)
    for (let i = 0; i < 10; i++) await Promise.resolve()
    h.scheduler.schedule('c', job('c'), true)
    for (let i = 0; i < 10; i++) await Promise.resolve()
    expect(calls).toEqual(['a', 'b']); expect(h.scheduler.stats.maxFrameMs).toBe(6)
    await h.run(); expect(calls).toEqual(['a', 'b', 'c'])
  })
})
