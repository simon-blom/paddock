import { afterEach, describe, expect, it, vi } from 'vitest'
import { WorkerRpc, type RpcPort } from './worker-rpc'
class Port implements RpcPort {
  onmessage: RpcPort['onmessage'] = null
  onerror: RpcPort['onerror'] = null
  onmessageerror: RpcPort['onmessageerror'] = null
  sent: { id: number; input: string }[] = []
  terminated = false
  postMessage(value: unknown) { this.sent.push(value as typeof this.sent[number]) }
  terminate() { this.terminated = true }
  reply(value: string) { this.onmessage?.({ data: { id: this.sent[this.sent.length - 1].id, value } } as MessageEvent) }
  start() { this.onmessage?.({ data: { id: this.sent[this.sent.length - 1].id, started: true } } as MessageEvent) }
}
const setup = (maxJobs = 32, maxBytes = 1024) => {
  const ports: Port[] = []
  const rpc = new WorkerRpc<string, string>(() => { const port = new Port(); ports.push(port); return port }, { maxJobs, maxBytes, timeoutMs: 100, idleMs: 500 })
  return { rpc, ports }
}
afterEach(() => vi.useRealTimers())
describe('renderer worker scheduling', () => {
  it('latest wins per owner, while one cancelled physical job remains active', async () => {
    const { rpc, ports } = setup(); const a = rpc.open(); const b = rpc.open()
    const first = rpc.submit(a, 'first', 10).catch(e => e.name)
    const second = rpc.submit(a, 'second', 10).catch(e => e.name)
    const other = rpc.submit(b, 'other', 10)
    const latest = rpc.submit(a, 'latest', 10)
    expect(await first).toBe('AbortError'); expect(await second).toBe('AbortError')
    expect(rpc.stats.active).toBe(1); expect(ports[0].sent.length).toBe(1)
    ports[0].reply('ignored'); expect(ports[0].sent[1].input).toBe('other')
    ports[0].reply('other'); expect(await other).toBe('other')
    ports[0].reply('latest'); expect(await latest).toBe('latest')
    rpc.close(a); rpc.close(b); expect(ports[0].terminated).toBe(true)
    expect(rpc.stats).toEqual({ workers: 0, active: 0, queued: 0, bytes: 0, owners: 0, waiting: 0, recoveries: 0 })
  })
  it('bounds jobs and retained bytes, including cancelled physical work', async () => {
    const { rpc, ports } = setup(2, 30); const a = rpc.open(); const b = rpc.open(); const c = rpc.open()
    const first = rpc.submit(a, 'first', 20).catch(e => e.name)
    rpc.cancel(a)
    await expect(rpc.submit(a, 'too many bytes', 20)).rejects.toThrow('budget')
    const second = rpc.submit(b, 'second', 10)
    await expect(rpc.submit(c, 'too many jobs', 0)).rejects.toThrow('budget')
    expect(rpc.stats.bytes).toBe(30); ports[0].reply('ignored'); ports[0].reply('second')
    expect(await first).toBe('AbortError'); expect(await second).toBe('second')
    rpc.close(a); rpc.close(b); rpc.close(c)
  })
  it('terminates timeouts, recovers the queue, and releases idle workers', async () => {
    vi.useFakeTimers()
    const { rpc, ports } = setup(); const a = rpc.open(); const b = rpc.open()
    const first = rpc.submit(a, 'stuck', 10).catch(e => e.message)
    const second = rpc.submit(b, 'next', 10)
    await vi.advanceTimersByTimeAsync(101)
    expect(await first).toContain('timed out'); expect(ports[0].terminated).toBe(true)
    expect(ports.length).toBe(2); ports[1].reply('next'); expect(await second).toBe('next')
    await vi.advanceTimersByTimeAsync(501); expect(ports[1].terminated).toBe(true)
    rpc.close(a); rpc.close(b)
  })
  it('release rejects both queued and active callers without orphaning promises', async () => {
    const { rpc, ports } = setup(); const a = rpc.open()
    const first = rpc.submit(a, 'active', 10).catch(e => e.name)
    const second = rpc.submit(a, 'queued', 10).catch(e => e.name)
    rpc.close(a)
    expect(await first).toBe('AbortError'); expect(await second).toBe('AbortError')
    expect(ports[0].terminated).toBe(true)
    await expect(rpc.submit(a, 'closed', 1)).rejects.toMatchObject({ name: 'AbortError' })
  })
  it('notifies deferred owners without retaining payloads, and cancels notifications on release', async () => {
    const { rpc, ports } = setup(1, 30); const a = rpc.open(); const b = rpc.open()
    const first = rpc.submit(a, 'first', 20)
    await expect(rpc.submit(b, 'later', 10)).rejects.toMatchObject({ name: 'RendererBusyError' })
    const retry = vi.fn()
    expect(rpc.whenAvailable(b, retry)).toBe(true)
    expect(rpc.stats.bytes).toBe(20); expect(rpc.stats.waiting).toBe(1)
    ports[0].reply('first'); await first; await Promise.resolve()
    expect(retry).toHaveBeenCalledTimes(1); expect(rpc.stats.waiting).toBe(0)
    rpc.whenAvailable(b, retry); rpc.close(b); await Promise.resolve()
    expect(retry).toHaveBeenCalledTimes(1); rpc.close(a)
  })
  it('does not spin microtasks when a physically active job leaves insufficient bytes', async () => {
    const { rpc, ports } = setup(32, 30); const a = rpc.open(); const b = rpc.open()
    const first = rpc.submit(a, 'first', 20)
    await expect(rpc.submit(b, 'later', 20)).rejects.toMatchObject({ name: 'RendererBusyError' })
    const retry = vi.fn()
    rpc.whenAvailable(b, retry); await Promise.resolve(); await Promise.resolve()
    expect(retry).not.toHaveBeenCalled()
    ports[0].reply('first'); await first; await Promise.resolve()
    expect(retry).toHaveBeenCalledTimes(1); rpc.close(a); rpc.close(b)
  })
  it('bounds deferred admission metadata as well as payload jobs', () => {
    const { rpc } = setup(); const owners = Array.from({ length: 513 }, () => rpc.open())
    for (const owner of owners.slice(0, 512)) expect(rpc.whenAvailable(owner, () => {})).toBe(true)
    expect(rpc.whenAvailable(owners[512], () => {})).toBe(false)
    expect(rpc.stats.waiting).toBe(512)
    for (const owner of owners) rpc.close(owner)
    expect(rpc.stats.waiting).toBe(0)
  })
  it('recovers an unstarted job once without dropping input or queued owners', async () => {
    vi.useFakeTimers()
    const ports: Port[] = []
    const rpc = new WorkerRpc<string, string>(() => { const port = new Port(); ports.push(port); return port },
      { maxJobs: 2, maxBytes: 30, timeoutMs: 1000, idleMs: 2000, startTimeoutMs: 100 })
    const a = rpc.open(); const b = rpc.open()
    const warm = rpc.submit(a, 'warm', 10); ports[0].reply('warm'); await warm
    const first = rpc.submit(a, 'retry-exact-input', 20)
    const next = rpc.submit(b, 'queued', 10)
    const staleReply = ports[0].onmessage!
    await vi.advanceTimersByTimeAsync(101)
    expect(ports[0].terminated).toBe(true); expect(ports[1].sent[0].input).toBe('retry-exact-input')
    expect(rpc.stats.bytes).toBe(30); expect(rpc.stats.recoveries).toBe(1)
    staleReply({ data: { id: ports[0].sent[1].id, value: 'stale' } } as MessageEvent)
    await vi.advanceTimersByTimeAsync(200)
    expect(ports).toHaveLength(2) // bootstrap/retry is not subject to the start watchdog
    ports[1].reply('fresh'); expect(await first).toBe('fresh')
    ports[1].start(); await vi.advanceTimersByTimeAsync(200)
    expect(ports).toHaveLength(2) // acknowledged slow work is not restarted
    ports[1].reply('queued'); expect(await next).toBe('queued')
    rpc.close(a); rpc.close(b)
  })
  it('a cancelled unstarted job is not replayed, and release cancels recovery', async () => {
    vi.useFakeTimers()
    const ports: Port[] = []
    const rpc = new WorkerRpc<string, string>(() => { const port = new Port(); ports.push(port); return port },
      { maxJobs: 4, maxBytes: 100, timeoutMs: 1000, idleMs: 2000, startTimeoutMs: 100 })
    const a = rpc.open()
    const warm = rpc.submit(a, 'warm', 10); ports[0].reply('warm'); await warm
    const old = rpc.submit(a, 'obsolete', 10).catch(e => e.name)
    const latest = rpc.submit(a, 'latest', 10)
    expect(await old).toBe('AbortError')
    await vi.advanceTimersByTimeAsync(101)
    expect(ports[1].sent[0].input).toBe('latest')
    ports[1].reply('latest'); expect(await latest).toBe('latest')
    const disposed = rpc.submit(a, 'disposed', 10).catch(e => e.name)
    rpc.close(a); await vi.advanceTimersByTimeAsync(2000)
    expect(await disposed).toBe('AbortError'); expect(ports).toHaveLength(2)
    expect(rpc.stats.active).toBe(0)
  })
  it('releases an idle parser with settled owners, but keeps active work alive', async () => {
    vi.useFakeTimers()
    const ports: Port[] = []
    const rpc = new WorkerRpc<string, string>(() => { const port = new Port(); ports.push(port); return port },
      { maxJobs: 4, maxBytes: 100, timeoutMs: 15000, idleMs: 5000, startTimeoutMs: 100 })
    const settled = rpc.open(); const stream = rpc.open()
    const first = rpc.submit(stream, 'one', 10); ports[0].reply('one'); await first
    await vi.advanceTimersByTimeAsync(4999)
    const active = rpc.submit(stream, 'two', 10); ports[0].start()
    await vi.advanceTimersByTimeAsync(5001)
    expect(ports[0].terminated).toBe(false) // active job keeps the compute deadline
    ports[0].reply('two'); expect(await active).toBe('two')
    rpc.close(stream); expect(rpc.stats.owners).toBe(1)
    await vi.advanceTimersByTimeAsync(5001)
    expect(ports[0].terminated).toBe(true)
    const reopened = rpc.submit(settled, 'fresh', 10)
    expect(ports).toHaveLength(2); ports[1].reply('fresh'); expect(await reopened).toBe('fresh')
    rpc.close(settled)
  })
})
