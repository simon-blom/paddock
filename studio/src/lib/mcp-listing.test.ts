import { beforeEach, afterEach, describe, it, expect, vi } from 'vitest'
import { createPinia, setActivePinia, disposePinia, getActivePinia } from 'pinia'
import { useMcpToolsStore, connectorKey } from '@/stores/mcpTools'
import { SEARCH_PROVIDERS } from './websearch'
import nativeSearchForm from '../../../apps/macos/Sources/PaddockUI/WebSearchSettingsView.swift?raw'

beforeEach(() => setActivePinia(createPinia()))
afterEach(() => { disposePinia(getActivePinia()!); vi.unstubAllGlobals() })
describe('shared native/web MCP inventory', () => {
  it('keeps cache reuse but rejects a stale in-flight result after a configuration change', async () => {
    const replies: ((value: Response) => void)[] = []
    const fetch = vi.fn(() => new Promise<Response>(resolve => replies.push(resolve)))
    vi.stubGlobal('fetch', fetch)
    const store = useMcpToolsStore(), key = connectorKey('fixture')
    store.ensureConnector('fixture'); store.ensureConnector('fixture')
    expect(fetch).toHaveBeenCalledTimes(1)
    store.invalidate(); store.ensureConnector('fixture')
    replies[1]!(new Response(JSON.stringify({ ok: true, tools: [{ name: 'new' }] })))
    await vi.waitFor(() => expect(store.get(key)?.tools[0]?.name).toBe('new'))
    replies[0]!(new Response(JSON.stringify({ ok: true, tools: [{ name: 'stale' }] })))
    await new Promise(resolve => setTimeout(resolve, 10))
    expect(store.get(key)?.tools[0]?.name).toBe('new')
  })
  it('retries a failed listing when reopened and does not treat it as an empty successful list', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response('{}', { status: 503 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true, tools: [] }))))
    const store = useMcpToolsStore(), key = connectorKey('fixture')
    store.ensureConnector('fixture')
    await vi.waitFor(() => expect(store.get(key)?.status).toBe('error'))
    store.ensureConnector('fixture')
    await vi.waitFor(() => expect(store.get(key)?.status).toBe('ok'))
  })
  it('keeps native web-search choices aligned with shared provider identifiers', () => {
    for (const provider of SEARCH_PROVIDERS) {
      expect(nativeSearchForm).toContain(`("${provider.id}", "${provider.label}")`)
      expect(nativeSearchForm).toContain(provider.keyUrl)
      expect(nativeSearchForm).toContain(provider.blurb)
    }
  })
})
