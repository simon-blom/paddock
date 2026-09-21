import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest'
import { createPinia, setActivePinia, disposePinia, getActivePinia } from 'pinia'
import { useModelsStore, type CloudEndpoint } from '@/stores/models'

const endpoint = (id: string, hasKey: boolean, allowUnauthenticated?: boolean): CloudEndpoint => ({
  id, name: id, kind: 'openai-compat', baseUrl: 'http://localhost:9991/v1', hasKey,
  allowUnauthenticated, createdAt: 1, models: [{ id: 'fixture/model', ctx: 8192 }],
})
let rows: CloudEndpoint[] = []
beforeEach(() => {
  setActivePinia(createPinia())
  const data = new Map<string, string>()
  vi.stubGlobal('localStorage', { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => data.set(key, value), removeItem: (key: string) => data.delete(key) })
  vi.stubGlobal('fetch', vi.fn(async (url: string) => new Response(JSON.stringify(url === '/api/cloud' ? rows : []), { status: 200 })))
})
afterEach(() => { disposePinia(getActivePinia()!); vi.unstubAllGlobals() })

describe('connections shared by web and native Studio', () => {
  it('admits only keyed or explicitly keyless endpoints into the composer', async () => {
    rows = [endpoint('keyed', true), endpoint('missing', false), endpoint('explicit', false, true), endpoint('disabled', false, false)]
    const models = useModelsStore()
    await models.refresh()
    expect(models.models.map(m => m.cloud?.endpoint)).toEqual(['keyed', 'explicit'])
  })
  it('retains exact provider identities after a store reload and never silently chooses auto-routing', async () => {
    const row = endpoint('router', true)
    row.baseUrl = 'https://openrouter.ai/api/v1'
    row.models = [{ id: 'fixture/model', provider: 'provider/turbo', ctx: 8192 }, { id: 'fixture/model', provider: 'provider/region', ctx: 16384 }]
    rows = [row]
    let models = useModelsStore()
    await models.refresh()
    const expected = ['cloud:router:fixture/model@provider/turbo', 'cloud:router:fixture/model@provider/region']
    expect(models.models.map(m => m.id)).toEqual(expected)
    disposePinia(getActivePinia()!)
    setActivePinia(createPinia())
    models = useModelsStore()
    await models.refresh()
    expect(models.models.map(m => m.id)).toEqual(expected)
    rows = []
    await models.refresh()
    expect(models.models).toHaveLength(0)
  })
})
