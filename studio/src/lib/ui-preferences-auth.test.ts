import { afterEach, describe, expect, it, vi } from 'vitest'
import { isKeyRequired, uiPreferences } from './ui-preferences'

// The first request the Studio makes is this read of its saved preferences,
// before anything mounts. From another machine a keyed manager refuses it
// until the browser unlocks, and the startup must tell that login apart from
// saved data that genuinely failed to open.
describe('opening the saved preferences', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('names a refusal for want of the key, so the key gate can show', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 401 })))
    const e = await uiPreferences.initialize().catch((err: unknown) => err)
    expect(isKeyRequired(e)).toBe(true)
  })

  it('keeps any other failure a failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 500 })))
    const e = await uiPreferences.initialize().catch((err: unknown) => err)
    expect(e).toBeInstanceOf(Error)
    expect(isKeyRequired(e)).toBe(false)
    expect(isKeyRequired(new Error('HTTP 401'))).toBe(false)
  })
})
