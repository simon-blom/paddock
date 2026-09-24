// Colours for a chart come off the design tokens, read at render time so the
// chart follows the theme like everything else on the page. Read them inside
// a computed keyed on the theme (HistoryChart's recipe): getComputedStyle is
// a forced style recalc and must not run per data tick.

export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

/** `#rgb` / `#rrggbb` / `rgb(...)` with an alpha applied; anything else is
 *  returned as-is. */
export function withAlpha(c: string, a: number): string {
  const s = c.trim()
  if (s.startsWith('#')) {
    let h = s.slice(1)
    if (h.length === 3)
      h = h
        .split('')
        .map((x) => x + x)
        .join('')
    const r = parseInt(h.slice(0, 2), 16)
    const g = parseInt(h.slice(2, 4), 16)
    const b = parseInt(h.slice(4, 6), 16)
    return `rgba(${r},${g},${b},${a})`
  }
  if (s.startsWith('rgb(')) return s.replace('rgb(', 'rgba(').replace(')', `,${a})`)
  return s
}
