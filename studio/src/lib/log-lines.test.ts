import { describe, expect, it } from 'vitest'
import fixture from './log-lines.fixture.json'
import { clean, parse, filterLogLines, type Level } from './log-lines'

describe('web/native log behavior contract', () => {
  it('cleans and parses the shared fixture with continuation severity', () => {
    let previous: Level | undefined
    for (const entry of fixture) {
      const line = parse(clean(entry.input), previous)
      expect(line.raw).toBe(entry.raw)
      expect(line.level ?? null).toBe(entry.level)
      expect(line.level ?? line.eff ?? null).toBe(entry.effective)
      expect(line.module ?? null).toBe(entry.module)
      expect(line.msg ?? null).toBe(entry.message)
      if (line.level) previous = line.level
    }
  })
  it('filters the complete retained history before the newest-1500 cap', () => {
    const lines = Array.from({ length: 4000 }, (_, i) => parse(`2026-09-16T09:12:04Z ${i % 2 ? 'WARN' : 'INFO'} runner: event ${i}`, undefined))
    expect(filterLogLines(lines, 'warn', '').length).toBe(1500)
    expect(filterLogLines(lines, 'all', ' Event 0 ')[0]?.msg).toBe('event 0')
    expect(filterLogLines(lines, 'error', '')).toEqual([])
  })
})
