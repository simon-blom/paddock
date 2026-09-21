import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useRegistryStore } from './registry'

vi.mock('@/lib/api', () => ({ gpuApi: {}, registryApi: {} }))
beforeEach(() => setActivePinia(createPinia()))
afterEach(() => vi.unstubAllGlobals())

it('a late estimate cannot overwrite the current settings or clear its pending state', async () => {
  const releases: ((r: Response) => void)[] = []
  const fetch = vi.fn((_input: RequestInfo | URL) => new Promise<Response>(resolve => releases.push(resolve)))
  vi.stubGlobal('fetch', fetch)
  const registry = useRegistryStore()
  const first = registry.estimate({ batch: 1 })
  const second = registry.estimate({ batch: 4, freeingPort: 11540, kv: 'f32', offloadRamGb: 8 })
  expect(fetch.mock.calls[1]?.[0]).toContain('freeing_port=11540')
  releases[0]!(Response.json({ models: { stale: {} } }))
  await first
  expect(registry.estimating).toBe(true)
  expect(registry.estimates).toEqual({})
  releases[1]!(Response.json({ models: { current: {} }, device: { unified: true } }))
  await second
  expect(registry.estimating).toBe(false)
  expect(registry.estimates).toEqual({ current: {} })
  expect(registry.estDevice?.unified).toBe(true)
})

it('HTTP failures leave the last successful estimate intact', async () => {
  const fetch = vi.fn().mockResolvedValueOnce(Response.json({ models: { current: {} } }))
    .mockResolvedValueOnce(Response.json({ error: 'fixture' }, { status: 503 }))
  vi.stubGlobal('fetch', fetch)
  const registry = useRegistryStore()
  await registry.estimate()
  await registry.estimate()
  expect(registry.estimates).toEqual({ current: {} })
  expect(registry.estimating).toBe(false)
})
