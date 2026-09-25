import { computed } from 'vue'
import { useSettingsStore } from '@/stores/settings'

type Theme = 'light' | 'dark'

/**
 * Light/dark theme: the setting is stored in the `settings` Pinia store
 * (persisted in SQLite); applying it writes `data-theme` on <html>.
 * Preferences hydrate before the app mounts.
 */
export function useTheme() {
  const settings = useSettingsStore()

  const theme = computed<Theme>({
    get: () => settings.theme,
    set: (v) => {
      settings.theme = v
    },
  })

  const setTheme = (newTheme: Theme) => {
    if (newTheme !== 'light' && newTheme !== 'dark') newTheme = 'dark'
    settings.theme = newTheme
    document.documentElement.setAttribute('data-theme', newTheme)
  }

  const toggleTheme = (): Theme => {
    const next: Theme = settings.theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    return next
  }

  const initTheme = () => {
    document.documentElement.setAttribute('data-theme', settings.theme)
  }

  return { theme, setTheme, toggleTheme, initTheme }
}
