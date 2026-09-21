import { fmtClock } from './format'

export type Level = 'TRACE' | 'DEBUG' | 'INFO' | 'WARN' | 'ERROR'
export interface Line {
  raw: string
  /** merged-mode source prefix: "manager" / "11540" */
  source?: string
  time?: string
  level?: Level
  /** filter-effective level for an unparsed line: inherited from the line
   *  above (continuations), so filtering to Warn+ hides banner art instead
   *  of showing it. Panic/backtrace text is promoted to a real ERROR. */
  eff?: Level
  module?: string
  msg?: string
}

// tracing's default format, with the merged stream's optional [source] prefix:
//   [11540] 2026-08-02T09:58:54.631489Z  INFO paddock_runner::drain: message
const RE =
  /^(?:\[([^\]]+)\]\s+)?(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z?)\s+(TRACE|DEBUG|INFO|WARN|ERROR)\s+([\w:.-]+):\s?(.*)$/

// A runner's stdout is its log file - the manager opens runner-<port>.log and
// hands over the handle at spawn - and tracing's terminal layer used to colour
// it regardless, so runner lines arrive wrapped in SGR escapes. The engine now
// only colours a real terminal, but every log already on disk carries them and
// .prev.log is never rewritten, so clean here as well. Without this the ESC
// bytes draw as missing-glyph boxes AND the line never matches RE at all: no
// clock, no level chip, and nothing for the level filter to select on.
// CSI (what tracing emits), then OSC, then the two-character escapes.
const ANSI = /\u001b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\u0007\u001b]*(?:\u0007|\u001b\\)|[@-Z\\-_])/g
// Anything that survives - a bare ESC, a stray C0 from a crashing child - is
// still an unprintable box. Tab stays; it is the only one that means something.
const CTRL = /[\u0000-\u0008\u000b-\u001f\u007f]/g
export function clean(raw: string): string {
  return raw.replace(ANSI, '').replace(CTRL, '')
}

// the log file speaks UTC; the viewer speaks the user's local clock, in
// fixed ISO 24h form (browser locales can't see the OS format preference)
function localTime(iso: string): string {
  const d = new Date(iso.endsWith('Z') ? iso : `${iso}Z`)
  return Number.isNaN(d.getTime()) ? iso.slice(11, 19) : fmtClock(d)
}

const PANIC = /panicked at|RUST_BACKTRACE|stack backtrace|^thread '/i

/** `lastLevel` threads the previous parsed line's level into continuations. */
export function parse(raw: string, lastLevel: Level | undefined): Line {
  const m = RE.exec(raw)
  if (!m) {
    if (PANIC.test(raw)) return { raw, level: 'ERROR' }
    return { raw, eff: lastLevel }
  }
  return { raw, source: m[1], time: localTime(m[2]), level: m[3] as Level, module: m[4], msg: m[5] }
}

export function filterLogLines(lines: readonly Line[], minimum: string, search: string): Line[] {
  const rank: Record<string, number> = { TRACE: 0, DEBUG: 1, INFO: 2, WARN: 3, ERROR: 4 }
  const min: Record<string, number> = { all: 0, info: 2, warn: 3, error: 4 }
  const q = search.trim().toLowerCase()
  return lines.filter(line => {
    const level = line.level ?? line.eff
    return (!(min[minimum] > 0) || (level !== undefined && rank[level] >= min[minimum]))
      && (!q || line.raw.toLowerCase().includes(q))
  }).slice(-1500)
}
