import { expect, it, vi } from 'vitest'
const wasm = vi.hoisted(() => ({ open: vi.fn() }))
vi.mock('@truespar/traverse-wasm', () => ({ TraverseDb: wasm }))
import { GraphSession } from './session'

it('closes a worker that finishes initialization after its pane was closed', async () => {
  const worker = { close: vi.fn() }
  let resolve!: (value: typeof worker) => void
  wasm.open.mockReturnValueOnce(new Promise(r => { resolve = r }))
  const session = new GraphSession(), pending = session.open()
  session.close()
  resolve(worker)
  await expect(pending).rejects.toThrow('Graph closed while opening')
  expect(worker.close).toHaveBeenCalledOnce()
  await expect(session.stats()).rejects.toThrow('graph session is closed')
})

it('never overwrites a newer worker when two opens finish out of order', async () => {
  const oldWorker = { close: vi.fn() }, newWorker = { close: vi.fn(), stats: vi.fn().mockResolvedValue({ nodes: 2, edges: 1 }) }
  let resolve!: (value: typeof oldWorker) => void
  wasm.open.mockReturnValueOnce(new Promise(r => { resolve = r })).mockResolvedValueOnce(newWorker)
  const session = new GraphSession(), old = session.open()
  await session.open()
  resolve(oldWorker)
  await expect(old).rejects.toThrow('Graph closed while opening')
  expect(oldWorker.close).toHaveBeenCalledOnce()
  expect(await session.stats()).toEqual({ nodes: 2, edges: 1 })
  expect(newWorker.close).not.toHaveBeenCalled()
  session.close()
  expect(newWorker.close).toHaveBeenCalledOnce()
})
