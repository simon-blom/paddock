import { describe, expect, it } from 'vitest'
import { READ_RUNS_KEEP, orderedRunQuestions, readTitle, withRun, type ReadRun } from './reads'

function run(at: number): ReadRun {
  return {
    at,
    model: 'DiffusionGemma 26B A4B',
    port: 11560,
    excerpt: 'Portal down',
    chars: 11,
    state: 'Portal down',
    fileName: '',
    questions: {},
    samples: 'auto',
    response: { answers: {}, diagnostics: { reads: 1, questions: [] } } as unknown as ReadRun['response'],
    ms: 10,
  }
}

describe('a read in the side panel', () => {
  it('honors native question and option order without losing rows', () => {
    const r = run(1)
    r.questions = { a: { type: 'noul', instructions: '' }, z: { type: 'choice', instructions: '', criteria: { apple: 'A', zebra: 'Z' } } }
    r.questionOrder = [['z', 'zebra', 'apple'], ['a']]
    const ordered = orderedRunQuestions(r)
    expect(Object.keys(ordered)).toEqual(['z', 'a'])
    expect(Object.keys(ordered.z!.criteria!)).toEqual(['zebra', 'apple'])
    r.questionOrder = [['z', 'zebra', 'apple']]
    expect(() => orderedRunQuestions(r)).toThrow('question order')
  })
  it('is named after its file, else the first line of its text', () => {
    expect(readTitle('Subject: Portal down\n\nHi', ' ticket.eml ')).toBe('ticket.eml')
    expect(readTitle('\n\n  Subject: Portal down again\r\nHi', '')).toBe('Subject: Portal down again')
    expect(readTitle('   ', '')).toBe('Untitled read')
    const long = readTitle('x'.repeat(200), '')
    expect(long.length).toBe(60)
    expect(long.endsWith('...')).toBe(true)
  })

  it('keeps its runs oldest first and drops the oldest past the cap', () => {
    let runs: ReadRun[] = []
    for (let i = 0; i < READ_RUNS_KEEP + 3; i++) runs = withRun(runs, run(i))
    expect(runs).toHaveLength(READ_RUNS_KEEP)
    expect(runs[0].at).toBe(3)
    expect(runs[runs.length - 1].at).toBe(READ_RUNS_KEEP + 2)
  })
})
