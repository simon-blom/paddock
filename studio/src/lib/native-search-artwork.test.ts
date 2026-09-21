import { describe, expect, it } from 'vitest'
import { siBrave, siPerplexity } from 'simple-icons'
import searchLogo from '../components/manage/SearchLogo.vue?raw'
import tavilyLogo from '../assets/tavily.svg?raw'
import nativeLogo from '../../../apps/macos/Sources/PaddockUI/SearchProviderLogo.swift?raw'
import { SEARCH_PROVIDERS } from './websearch'

const nativeAssets = import.meta.glob<string>(
  '../../../apps/macos/Sources/PaddockUI/Resources/{Brave,Exa,Firecrawl,Perplexity,Tavily}.svg',
  { query: '?raw', import: 'default', eager: true },
)
const asset = (name: string) => {
  const svg = nativeAssets[`../../../apps/macos/Sources/PaddockUI/Resources/${name}.svg`]
  if (svg === undefined) throw new Error(`Missing native provider artwork: ${name}`)
  return svg
}

describe('native/web search artwork parity', () => {
  it('shares all five provider identities and the actual web SVG geometry', () => {
    for (const { id, label } of SEARCH_PROVIDERS) {
      expect(nativeLogo).toContain(`case .${id}: "${label}"`)
    }
    for (const name of ['Exa', 'Firecrawl']) {
      const path = asset(name).match(/\sd="([^"]+)"/)?.[1]
      expect(path).toBeTruthy()
      expect(searchLogo).toContain(`d="${path}"`)
    }
    // A Windows checkout can hand one copy back with CRLF; same artwork either way.
    const lf = (svg: string) => svg.replace(/\r\n/g, '\n').trim()
    expect(lf(asset('Tavily'))).toBe(lf(tavilyLogo))
    for (const [name, icon] of [['Brave', siBrave], ['Perplexity', siPerplexity]] as const) {
      expect(asset(name)).toContain(`d="${icon.path}"`)
      expect(nativeLogo.toLowerCase()).toContain(`0x${icon.hex.toLowerCase()}`)
    }
  })

  it('keeps the web theme colours and non-square flame dimensions', () => {
    for (const hex of ['1f40ed', '6f88ff', 'fa5d19']) {
      expect(searchLogo.toLowerCase()).toContain(`#${hex}`)
      expect(nativeLogo.toLowerCase()).toContain(`0x${hex}`)
    }
    expect(searchLogo).toContain('(size * 200) / 284')
    expect(nativeLogo).toContain('200.0 / 284.0')
    expect(nativeLogo).toContain('case .tavily: nil')
    expect(nativeLogo).not.toContain('.saturation(0)')
  })
})
