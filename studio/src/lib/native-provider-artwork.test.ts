import { describe, expect, it } from 'vitest'
import vendorLogo from '../components/manage/VendorLogo.vue?raw'
import prismLogo from '../../../apps/macos/Sources/PaddockUI/Resources/PrismML.svg?raw'
import nativeArtwork from '../../../apps/macos/Sources/PaddockUI/ProviderArtwork.swift?raw'

describe('Prism ML provider artwork parity', () => {
  it('shares the official emblem geometry between Studio and macOS', () => {
    const webMark = vendorLogo.match(/<svg\s+v-else-if="vendor === 'Prism ML'"[\s\S]*?<\/svg>/)?.[0]
    expect(webMark).toBeDefined()
    const paths = (svg: string) => [...svg.matchAll(/\sd="([^"]+)"/g)].map((match) => match[1])
    expect(paths(prismLogo)).toHaveLength(4)
    expect(paths(webMark!)).toEqual(paths(prismLogo))
    const viewBox = prismLogo.match(/viewBox="([^"]+)"/)?.[1]
    expect(viewBox).toBe('0 0 36.2855 29.7604')
    expect(webMark).toContain(`viewBox="${viewBox}"`)
    expect(webMark).toContain('fill="currentColor"')
    expect(webMark).toContain('aria-label="Prism ML"')
  })

  it('bundles the native mark locally under the registry vendor name', () => {
    expect(nativeArtwork).toContain('"Prism ML": "PrismML"')
    expect(prismLogo).not.toMatch(/<(?:script|image|foreignObject)\b|\bhref\s*=/i)
  })
})
