import { describe, expect, it, vi } from 'vitest'
import wasmCacheSource from '../../vendor/lector/core/dist/worker/wasm-cache-IXSAKVDT.js?raw'
import traverseSource from '../../vendor/traverse/wasm/worker.js?raw'
import lectorSource from '../../vendor/lector/core/dist/index.js?raw'
import { LectorEngine } from '@truespar/lector-core'
import { createPreferences } from './ui-preferences'

function wasmLoader(fetcher: typeof fetch) {
  // Run the exact vendored worker code with real WebAssembly and no browser
  // storage globals. Imports here only provide bundler scaffolding.
  const source = wasmCacheSource.replace(/^import .*;$/gm, '').replace(/^export .*;$/gm, '')
  return new Function('fetch', 'WebAssembly', `${source}\nreturn loadWasmCached`)(fetcher, WebAssembly) as
    (url: string, imports: WebAssembly.Imports) => Promise<WebAssembly.WebAssemblyInstantiatedSource>
}
const emptyModule = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])
const response = () => new Response(emptyModule, { headers: { 'Content-Type': 'application/wasm' } })

describe('embedded viewers without browser persistence', () => {
  it('coalesces WASM compilation but creates independent instances in memory', async () => {
    const fetcher = vi.fn(async () => response()), load = wasmLoader(fetcher)
    const [a, b] = await Promise.all([load('/pdf.wasm', {}), load('/pdf.wasm', {})])
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(a.module).toBe(b.module); expect(a.instance).not.toBe(b.instance)
  })
  it('retries failed WASM loads and bounds the compiled-module cache', async () => {
    const fetcher = vi.fn(async () => response()), load = wasmLoader(fetcher)
    fetcher.mockResolvedValueOnce(new Response('', { status: 503 }))
    await expect(load('/pdf.wasm', {})).rejects.toThrow('503')
    await load('/pdf.wasm', {})
    for (let n = 0; n < 4; n++) await load(`/other-${n}.wasm`, {})
    await load('/pdf.wasm', {})
    expect(fetcher).toHaveBeenCalledTimes(7)
  })
  it('falls back for incorrect WASM MIME types without any persistent cache', async () => {
    const load = wasmLoader(vi.fn(async () => new Response(emptyModule)))
    expect((await load('/pdf.wasm', {})).module).toBeInstanceOf(WebAssembly.Module)
  })
  it('routes Lector preferences to the host and makes unintegrated instances ephemeral', () => {
    const prefs = createPreferences({ load: async () => ({}), importMissing: async () => ({}), save: async () => {} })
    const engine = new LectorEngine({ wasmUrl: '/pdf.wasm', wasmJsUrl: '/pdf.js', preferenceStore: prefs })
    expect(engine.preferenceStore).toBe(prefs)
    expect(lectorSource).not.toMatch(/localStorage|sessionStorage|indexedDB/)
    expect(lectorSource).toContain('createPreferenceStore(storageKey, maxRecent, engineAny.preferenceStore)')
  })
  it('cannot open a second persistent graph inside the browser worker', () => {
    expect(traverseSource).not.toMatch(/getDirectory|createSyncAccessHandle|indexedDB|localStorage/)
    expect(traverseSource).toContain("throw new Error('Use the Paddock graph API for persistence')")
    // Import/export remain available: graph bytes still round-trip via Rust.
    expect(traverseSource).toContain("case 'exportTvdb'")
    expect(traverseSource).toContain("case 'loadTvdb'")
  })
})
