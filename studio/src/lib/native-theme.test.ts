import { describe, expect, it } from 'vitest'
import palette from '../../../apps/macos/Sources/PaddockDesign/Resources/appearance.json'
import { hslChannels, nativeThemeCSS } from '../../native-workspace/theme'

describe('shared native appearance contract', () => {
  it('defines every native color in both schemes and never aliases two roles accidentally', () => {
    const names = new Set<string>()
    for (const pair of Object.values(palette.colors)) {
      expect(pair.light).toMatch(/^#[0-9a-f]{6}([0-9a-f]{2})?$/)
      expect(pair.dark).toMatch(/^#[0-9a-f]{6}([0-9a-f]{2})?$/)
      for (const name of pair.web) {
        expect(names.has(name)).toBe(false)
        names.add(name)
      }
    }
    for (const name of ['accent', 'accent-text', 'accent-hover', 'accent-active', 'accent-subtle', 'border-focus', 'document-stage']) {
      expect(names.has(name)).toBe(true)
    }
  })
  it('keeps chrome neutral without removing semantic status colors', () => {
    for (const [role, pair] of Object.entries(palette.colors)) {
      for (const scheme of ['light', 'dark'] as const) {
        const hex = pair[scheme]
        if (/^(caution|error|success)/.test(role)) continue
        expect(hex.slice(1, 3)).toBe(hex.slice(3, 5))
        expect(hex.slice(3, 5)).toBe(hex.slice(5, 7))
      }
    }
    expect(palette.colors.error.dark).not.toBe(palette.colors.primary.dark)
  })
  it('derives Markdown HSL tokens and radius scales from the same palette', () => {
    expect(hslChannels('#ffffff')).toBe('0.000 0.000% 100.000%')
    expect(hslChannels('#ff0000')).toBe('0.000 100.000% 50.000%')
    expect(hslChannels('#00ff00')).toBe('120.000 100.000% 50.000%')
    const css = nativeThemeCSS()
    for (const theme of ['light', 'dark'] as const) {
      expect(css).toContain(`[data-theme="${theme}"]`)
      expect(css).toContain(`--ms-ring:${hslChannels(palette.colors.focus[theme])}`)
      expect(css).toContain(`--pk-document-stage:${palette.colors.surface[theme]}`)
    }
    expect(css).toContain(`--pk-radius-md:${palette.radii.md}px`)
    expect(css).toContain('data-native-theme')
  })
})
