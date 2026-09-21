import palette from '../../apps/macos/Sources/PaddockDesign/Resources/appearance.json'

// Swift bundles this exact resource. No second hand-copied native/web palette.
// Only the native entry point installs it; the standalone web Studio keeps its
// own theme while sharing the same viewer-to-host token adapters.
const markdownRoles = {
  background: 'canvas', foreground: 'primary', muted: 'surface',
  'muted-foreground': 'secondary', secondary: 'elevated',
  'secondary-foreground': 'primary', accent: 'elevated',
  'accent-foreground': 'primary', primary: 'accent',
  'primary-foreground': 'actionForeground', destructive: 'error',
  border: 'border', ring: 'focus', popover: 'popup',
  'popover-foreground': 'primary', info: 'info', success: 'success', warning: 'caution',
} as const

/** Markstream consumes HSL channels, not a CSS color. Derive them from the
 * palette instead of keeping yet another set of literals in its adapter. */
export function hslChannels(hex: string): string {
  const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255) as [number, number, number]
  const max = Math.max(r, g, b), min = Math.min(r, g, b), delta = max - min
  const l = (max + min) / 2
  const s = delta === 0 ? 0 : delta / (1 - Math.abs(2 * l - 1))
  let h = delta === 0 ? 0 : max === r ? ((g - b) / delta) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4
  h = (h * 60 + 360) % 360
  return `${h.toFixed(3)} ${(s * 100).toFixed(3)}% ${(l * 100).toFixed(3)}%`
}

export function nativeThemeCSS(): string {
  return (['light', 'dark'] as const).map(theme => {
    const root = `:root[data-native-theme][data-theme="${theme}"]`
    const colors = Object.values(palette.colors).flatMap(pair => pair.web.map(name => `--pk-${name}:${pair[theme]};`)).join('')
    const radii = Object.entries(palette.radii).map(([name, value]) => `--pk-radius-${name}:${value}px;`).join('')
    const markdown = Object.entries(markdownRoles).map(([name, role]) => `--ms-${name}:${hslChannels(palette.colors[role][theme])};`).join('')
    return `${root}{${colors}${radii}}\n${root} .markstream-vue{${markdown}}`
  }).join('\n')
}

export function installNativeTheme(): void {
  document.documentElement.dataset.nativeTheme = ''
  if (!document.documentElement.dataset.theme) {
    document.documentElement.dataset.theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  const style = document.createElement('style')
  style.id = 'paddock-native-theme'
  style.textContent = nativeThemeCSS()
  document.head.append(style)
}
