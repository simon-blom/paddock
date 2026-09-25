import { describe, expect, it } from 'vitest'
import {
  answerNames,
  choiceBars,
  cleanId,
  confidenceBin,
  curlFor,
  deriveId,
  dropReferences,
  duplicateQuestion,
  entropyWord,
  followRename,
  fromWire,
  imageRef,
  keptImages,
  nearTie,
  newQuestion,
  noulBars,
  parseQuestionsJson,
  requestBody,
  routeError,
  scoreBars,
  scorePosition,
  toWire,
  validate,
  thoughtsOf,
  type ReadAnswerChoice,
  type ReadAnswerScore,
  type ReadResponse,
  type ReadRun,
} from './reads'

function choice(id: string, names: string[]): ReturnType<typeof newQuestion> {
  const q = newQuestion('choice')
  q.id = id
  q.instructions = 'Which one?'
  q.options = names.map((name) => ({ name, description: '' }))
  return q
}

describe('ids', () => {
  it('derives a content-word id from the instructions', () => {
    expect(deriveId('Is the customer angry?')).toBe('customer_angry')
    expect(deriveId('Rate the urgency of this ticket')).toBe('urgency_ticket')
    expect(deriveId('Which department should handle this?')).toBe('department_handle')
  })
  it('falls back to the question\'s last words when only stop words remain, and stays unique', () => {
    expect(deriveId('Is it?')).toBe('is_it')
    expect(deriveId('What is this message about?')).toBe('message_about')
    expect(deriveId('')).toBe('q')
    expect(deriveId('Is the customer angry?', ['customer_angry'])).toBe('customer_angry_2')
    expect(deriveId('Is the customer angry?', ['customer_angry', 'customer_angry_2'])).toBe(
      'customer_angry_3',
    )
  })
  it('cleans a hand-typed id to the runner rule', () => {
    expect(cleanId('two words: here')).toBe('two_words_here')
  })
})

describe('validate', () => {
  it('mirrors the runner: ids, duplicates, option and level sets', () => {
    const a = choice('a', ['x', 'y'])
    const b = choice('a', ['x', 'x'])
    const c = newQuestion('score')
    c.id = 'bad id'
    c.levels = ['low']
    const v = validate([a, b, c])
    expect(v.ok).toBe(false)
    expect(v.rows[a.key]).toBeUndefined()
    expect(v.rows[b.key]).toMatch(/already has the id/)
    expect(v.rows[c.key]).toMatch(/one word/)
    const d = choice('d', ['x', 'x'])
    expect(validate([d]).rows[d.key]).toMatch(/must differ/)
    const e = newQuestion('score')
    e.id = 'e'
    e.levels = ['low']
    expect(validate([e]).rows[e.key]).toMatch(/at least two levels/)
  })
  it('caps the set at the advertised maximum', () => {
    const qs = Array.from({ length: 3 }, (_, i) => choice(`q${i}`, ['x', 'y']))
    const v = validate(qs, { canvasWidth: 256, maxQuestions: 2, maxSamples: 32, maxSteps: 1, images: false, conditional: false, think: false, types: [] })
    expect(v.set[0]).toMatch(/up to 2 per call/)
    expect(validate([]).set[0]).toMatch(/at least one/)
  })
})

describe('wire form', () => {
  it('serialises each type the way the runner reads it and round-trips', () => {
    const yn = newQuestion('noul')
    yn.id = 'angry'
    yn.instructions = 'Is the customer angry?'
    yn.yesMeans = 'clear anger'
    const ch = choice('dept', ['sales', 'support'])
    ch.options[1].description = 'a broken thing'
    const sc = newQuestion('score')
    sc.id = 'urgency'
    sc.instructions = 'How urgent?'
    sc.levels = ['low', 'medium', 'high']
    const wire = toWire([yn, ch, sc])
    expect(Object.keys(wire)).toEqual(['angry', 'dept', 'urgency'])
    expect(wire.angry).toEqual({
      type: 'noul',
      instructions: 'Is the customer angry?',
      criteria: { true: 'clear anger' },
    })
    expect(wire.dept.criteria).toEqual({ sales: '', support: 'a broken thing' })
    expect(wire.urgency.criteria).toEqual(['low', 'medium', 'high'])

    const back = fromWire(wire)
    expect(back.errors).toEqual([])
    expect(back.questions.map((q) => [q.id, q.type])).toEqual([
      ['angry', 'noul'],
      ['dept', 'choice'],
      ['urgency', 'score'],
    ])
    expect(back.questions[0].yesMeans).toBe('clear anger')
    expect(back.questions[1].options).toEqual([
      { name: 'sales', description: '' },
      { name: 'support', description: 'a broken thing' },
    ])
    expect(back.questions[2].levels).toEqual(['low', 'medium', 'high'])
    expect(toWire(back.questions)).toEqual(wire)
  })
  it('unwraps a whole request body and reports bad rows without dropping good ones', () => {
    const p = parseQuestionsJson(
      JSON.stringify({
        state: 'hello',
        samples: 4,
        questions: {
          ok: { type: 'bool', instructions: 'x' },
          weird: { type: 'rank', instructions: 'y' },
          list: [1, 2],
        },
      }),
    )
    expect(p.state).toBe('hello')
    expect(p.samples).toBe(4)
    expect(p.questions.map((q) => q.id)).toEqual(['ok'])
    expect(p.questions[0].type).toBe('noul')
    expect(p.errors).toHaveLength(2)
    expect(parseQuestionsJson('{').errors).toHaveLength(1)
    expect(parseQuestionsJson('[]').errors[0]).toMatch(/object/)
  })
  it('keeps a type switch lossless: the other criteria stay on the row', () => {
    const q = choice('q', ['a', 'b'])
    q.type = 'score'
    q.levels = ['1', '2']
    q.type = 'choice'
    expect(q.options.map((o) => o.name)).toEqual(['a', 'b'])
    const d = duplicateQuestion(q, ['q'])
    expect(d.id).toBe('q_2')
    expect(d.key).not.toBe(q.key)
    d.options[0].name = 'changed'
    expect(q.options[0].name).toBe('a')
  })
  it('sends samples only when not auto, and the curl carries the body', () => {
    const q = choice('q', ['a', 'b'])
    expect(requestBody('s', [q], 'auto')).toEqual({ state: 's', questions: toWire([q]) })
    expect(requestBody('s', [q], 3).samples).toBe(3)
    const c = curlFor(11560, requestBody("it's", [q], 'auto'))
    expect(c).toContain('http://localhost:11560/v1/systemone')
    expect(c).toContain("<<'JSON'")
    expect(c).toContain('"state": "it\'s"')
  })
})

describe('answers', () => {
  const ch: ReadAnswerChoice = {
    type: 'choice',
    choice: 'support',
    probabilities: { sales: 0.2, support: 0.7, billing: 0.1 },
    confidence: 0.7,
    agreement: 1,
    outside: 0.03,
  }
  it('ranks choice bars with the winner marked and the outside mass last', () => {
    const bars = choiceBars(ch)
    expect(bars.map((b) => b.name)).toEqual(['support', 'sales', 'billing', 'outside the options'])
    expect(bars[0].role).toBe('winner')
    expect(bars[3].role).toBe('outside')
    expect(bars[3].p).toBeCloseTo(0.03)
    expect(nearTie(bars)).toBe(false)
    expect(nearTie(choiceBars({ ...ch, probabilities: { a: 0.48, b: 0.46, c: 0.06 } }))).toBe(true)
  })
  it('keeps score bars in level order and places the marker', () => {
    const sc: ReadAnswerScore = {
      type: 'score',
      score: 1.4,
      level: 'medium',
      legend: { '0': 'low', '1': 'medium', '2': 'high' },
      probabilities: { '0': 0.1, '1': 0.4, '2': 0.5 },
      confidence: 0.5,
      agreement: 0.75,
      outside: 0,
    }
    expect(scoreBars(sc).map((b) => b.name)).toEqual(['low', 'medium', 'high', 'outside the options'])
    expect(scorePosition(sc)).toBeCloseTo(0.7)
    expect(noulBars({ type: 'noul', noul: 0.2, confidence: 0.8, agreement: 1, outside: 0 })[1].role).toBe(
      'winner',
    )
  })
  it('bins entropy and confidence at visible edges', () => {
    expect(entropyWord(0.01)).toBe('settled')
    expect(entropyWord(0.3)).toBe('unsettled')
    expect(entropyWord(0.8)).toBe('split')
    expect(entropyWord(Number.NaN)).toBe('split')
    expect([0.2, 0.5, 0.7, 0.95].map(confidenceBin)).toEqual([0, 1, 2, 3])
  })
})

describe('error routing', () => {
  it('pins a refusal to the question it names, the window to the state', () => {
    expect(routeError('question "dept": label "AB" is not a single token in the answer template')).toEqual({
      where: 'row',
      id: 'dept',
    })
    expect(routeError('the prompt is 9000 tokens and the canvas 32; the window is 8192')).toEqual({
      where: 'state',
    })
    expect(routeError('questions: 70 given, at most 64 per request')).toEqual({ where: 'questions' })
    expect(routeError('no chat model is loaded')).toEqual({ where: 'page' })
  })
})

describe('conditions', () => {
  function staged() {
    const kind = choice('kind', ['bug', 'question'])
    const severity = newQuestion('score')
    severity.id = 'severity'
    severity.instructions = 'How severe?'
    severity.levels = ['minor', 'major', 'critical']
    severity.askIf = [{ key: kind.key, values: ['bug'] }]
    const page = newQuestion('noul')
    page.id = 'page_oncall'
    page.instructions = 'Page on-call?'
    page.after = [severity.key]
    page.alone = true
    return { kind, severity, page, qs: [kind, severity, page] }
  }

  it('sends ask_if, depends_on and alone by id, and reads them back by row', () => {
    const { qs } = staged()
    const w = toWire(qs)
    expect(w.severity.ask_if).toEqual({ kind: ['bug'] })
    expect(w.severity.depends_on).toBeUndefined()
    expect(w.page_oncall.depends_on).toEqual(['severity'])
    expect(w.page_oncall.alone).toBe(true)
    expect(w.kind.ask_if).toBeUndefined()
    const back = fromWire({ questions: w })
    expect(back.errors).toEqual([])
    const [k, sv, pg] = back.questions
    expect(sv.askIf).toEqual([{ key: k.key, values: ['bug'] }])
    expect(pg.after).toEqual([sv.key])
    expect(pg.alone).toBe(true)
    expect(toWire(back.questions)).toEqual(w)
  })

  it('resolves a condition on a question written after it, and drops one on no question', () => {
    const p = fromWire({
      questions: {
        a: { type: 'noul', instructions: 'A?', ask_if: { b: ['yes'] }, depends_on: ['b', 'nope'] },
        b: { type: 'noul', instructions: 'B?' },
      },
    })
    const [a, b] = p.questions
    expect(a.askIf).toEqual([{ key: b.key, values: ['yes'] }])
    // ask_if's question is not repeated in `after`
    expect(a.after).toEqual([])
    expect(p.errors).toEqual(['a: its condition names "nope", which is not a question here.'])
  })

  it('refuses answers a question cannot give, empty picks, loops, and a model without conditions', () => {
    const { kind, severity, page, qs } = staged()
    expect(validate(qs).ok).toBe(true)
    severity.askIf = [{ key: kind.key, values: ['feature'] }]
    expect(validate(qs).rows[severity.key]).toBe('"kind" cannot answer "feature".')
    severity.askIf = [{ key: kind.key, values: [] }]
    expect(validate(qs).rows[severity.key]).toMatch(/Pick the answers of "kind"/)
    severity.askIf = [{ key: page.key, values: ['yes'] }]
    expect(validate(qs).set).toContain('These questions wait on each other: severity, page_oncall.')
    severity.askIf = []
    const old = { canvasWidth: 256, maxQuestions: 64, maxSamples: 32, maxSteps: 1, images: false, conditional: false, think: false, types: [] }
    expect(validate(qs, old).set[0]).toMatch(/conditions need a newer runner/)
  })

  it('follows an option renamed in place, and forgets a removed question', () => {
    const { kind, severity, page, qs } = staged()
    const before = answerNames(kind)
    kind.options[0].name = 'defect'
    followRename(qs, kind.key, before, answerNames(kind))
    expect(severity.askIf[0].values).toEqual(['defect'])
    expect(answerNames(page)).toEqual(['yes', 'no'])
    qs.splice(1, 1)
    dropReferences(qs, severity.key)
    expect(page.after).toEqual([])
  })

  it('copies conditions when a row is duplicated, without sharing them', () => {
    const { severity, qs } = staged()
    const copy = duplicateQuestion(severity, qs.map((q) => q.id))
    copy.askIf[0].values.push('question')
    expect(severity.askIf[0].values).toEqual(['bug'])
  })
})

describe('read settings and pictures', () => {
  it('sends steps, think and images only when they are set', () => {
    const q = choice('topic', ['a', 'b'])
    expect(requestBody('t', [q], 'auto', { steps: 1, think: 0, images: [] })).toEqual({
      state: 't',
      questions: toWire([q]),
    })
    const b = requestBody('t', [q], 2, { steps: 4, think: 256, images: ['data:image/png;base64,AA=='] })
    expect(b.steps).toBe(4)
    expect(b.think).toBe(256)
    expect(b.images).toEqual(['data:image/png;base64,AA=='])
    const p = fromWire(JSON.parse(JSON.stringify(b)))
    expect(p.steps).toBe(4)
    expect(p.think).toBe(256)
  })

  it('writes the multipart curl with the files named, never the base64', () => {
    const q = choice('topic', ['a', 'b'])
    const b = requestBody('t', [q], 'auto', { images: ['data:image/png;base64,QUFBQQ=='] })
    const c = curlFor(8080, b, ['my "photo";1.png'])
    expect(c).not.toContain('QUFBQQ')
    expect(c).toContain('-F "request=<request.json;type=application/json"')
    expect(c).toContain('-F "image=@my _photo__1.png"')
    expect(c.trimEnd().endsWith('\\')).toBe(false)
  })

  it('keys a picture by its bytes and keeps only the pictures a run still uses', () => {
    const a = 'data:image/png;base64,AAAA'
    const b = 'data:image/png;base64,BBBB'
    expect(imageRef(a)).toBe(imageRef(a))
    expect(imageRef(a)).not.toBe(imageRef(b))
    const run = (refs: string[]): ReadRun =>
      ({ images: refs.map((ref) => ({ name: 'x.png', ref })) }) as unknown as ReadRun
    const ra = imageRef(a)
    const rb = imageRef(b)
    expect(keptImages([run([ra])], { [rb]: b }, { [ra]: a })).toEqual({ [ra]: a })
    expect(keptImages([run([])], { [ra]: a }, {})).toBeUndefined()
  })

  it('lists a response\'s thoughts, one or many', () => {
    const r = (thought: unknown) => ({ diagnostics: { thought } }) as unknown as ReadResponse
    const t = { text: 'x', tokens: 1, closed: true, ms: 1 }
    expect(thoughtsOf(r(null))).toEqual([])
    expect(thoughtsOf(r(t))).toEqual([t])
    expect(thoughtsOf(r([t, t]))).toHaveLength(2)
  })
})
