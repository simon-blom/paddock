// Structured reads - the pure half of the Studio's /v1/systemone client: the
// question model the editor works on, its wire form, the validation the
// runner would otherwise answer with a 422, and the readings the Answers
// card derives from a response. No DOM and no store here, so all of it is
// testable as data (reads.test.ts). The wire shape is the one the runner
// speaks (crates/paddock-runner/src/systemone.rs) - Jev's request and
// answer forms plus the runner's own `outside` / `agreement` / diagnostics.
import { uuid } from '@/lib/uuid'

export type ReadType = 'noul' | 'choice' | 'score'
/** `"auto"` reads once and re-reads while a slot is unsettled (the runner's
 *  entropy threshold, up to four in all); a number reads exactly that often. */
export type Samples = 'auto' | number

export interface ReadOption {
  name: string
  description: string
}

/** One question as the editor holds it. Every type's criteria live on the
 *  row at once, so switching the type keeps what was typed for the others
 *  and a mis-click costs nothing (SurveyJS documents the data loss its own
 *  type switch causes). `key` is the row's identity in the editor; `id` is
 *  what goes on the wire, derived from the text until the user edits it. */
export interface ReadQuestion {
  key: string
  id: string
  idTouched: boolean
  type: ReadType
  instructions: string
  yesMeans: string
  noMeans: string
  options: ReadOption[]
  levels: string[]
}

/** What a block-diffusion endpoint advertises about reads (`/api/server`
 *  `structured_read`): one canvas holds `canvasWidth` positions, and the
 *  per-call caps are the runner's. */
export interface StructuredReadCaps {
  canvasWidth: number
  maxQuestions: number
  maxSamples: number
  types: string[]
}

/** The runner's own limits, used when a cap is not advertised. */
export const DEFAULT_MAX_QUESTIONS = 64
export const DEFAULT_MAX_SAMPLES = 32

export const READ_TYPES: { value: ReadType; label: string }[] = [
  { value: 'noul', label: 'Yes / no' },
  { value: 'choice', label: 'Choice' },
  { value: 'score', label: 'Score' },
]

export function newQuestion(type: ReadType = 'noul'): ReadQuestion {
  return {
    key: uuid(),
    id: '',
    idTouched: false,
    type,
    instructions: '',
    yesMeans: '',
    noMeans: '',
    options: [
      { name: '', description: '' },
      { name: '', description: '' },
    ],
    levels: ['', '', ''],
  }
}

export function duplicateQuestion(q: ReadQuestion, taken: Iterable<string>): ReadQuestion {
  return {
    ...q,
    key: uuid(),
    id: uniqueId(q.id || 'q', taken),
    idTouched: true,
    options: q.options.map((o) => ({ ...o })),
    levels: [...q.levels],
  }
}

// ── ids ──────────────────────────────────────────────────────────────────────

// Words that carry no meaning in a question's id. "Is the customer angry?"
// should become `customer_angry`, not `is_the_customer`.
const STOP = new Set(
  (
    'a an the is are was were be been being do does did of to in on at for with and or ' +
    'this that these those it its what which how who whom there here any from by as into ' +
    'about than then text state message following above below please rate decide whether ' +
    'if has have had will would should could can not no yes you your we our they their he ' +
    'she him her his them i me my one'
  ).split(' '),
)

/** A single-word id from the instructions: up to three content words joined
 *  with `_`, lowercase, unique against `taken`. */
export function deriveId(instructions: string, taken: Iterable<string> = []): string {
  const words = instructions
    .toLowerCase()
    .replace(/[^a-z0-9\s_-]+/g, ' ')
    .split(/\s+/)
    .filter(Boolean)
  let pick = words.filter((w) => !STOP.has(w)).slice(0, 3)
  // nothing but stop words ("What is this message about?"): a question ends
  // on its object, so the last two words name it better than the first two
  if (!pick.length) pick = words.slice(-2)
  let base = pick.join('_').replace(/^[_-]+|[_-]+$/g, '')
  if (base.length > 32) base = base.slice(0, 32).replace(/[_-]+$/, '')
  return uniqueId(base || 'q', taken)
}

/** `base`, or `base_2`, `base_3`... - the first not in `taken`. */
export function uniqueId(base: string, taken: Iterable<string>): string {
  const used = new Set(taken)
  if (!used.has(base)) return base
  for (let n = 2; ; n++) {
    const c = `${base}_${n}`
    if (!used.has(c)) return c
  }
}

/** The runner's rule for a hand-typed id: one word, no `:` - it is written
 *  into the answer template as `id: label`. */
export function cleanId(raw: string): string {
  return raw.replace(/[\s:]+/g, '_')
}

// ── validation (mirrors the runner's 422s, so they are seen before the run) ─

export interface Validation {
  /** first problem per row, keyed by the row's `key` */
  rows: Record<string, string>
  /** problems with the set as a whole */
  set: string[]
  ok: boolean
}

export function validate(qs: ReadQuestion[], caps?: StructuredReadCaps | null): Validation {
  const rows: Record<string, string> = {}
  const set: string[] = []
  if (!qs.length) set.push('Add at least one question.')
  const maxQ = caps?.maxQuestions ?? DEFAULT_MAX_QUESTIONS
  if (qs.length > maxQ) set.push(`${qs.length} questions; this model reads up to ${maxQ} per call.`)
  const seen = new Set<string>()
  for (const q of qs) {
    const id = q.id.trim()
    let err: string | undefined
    if (!id) err = 'Give the question an id - one word.'
    else if (/[\s:]/.test(id)) err = 'The id is one word without ":" - it is written into the answer template.'
    else if (seen.has(id)) err = `Another question already has the id "${id}".`
    else seen.add(id)
    if (!err && q.type === 'choice') {
      const names = q.options.map((o) => o.name.trim())
      if (names.length < 2) err = 'A choice needs at least two options.'
      else if (names.some((n) => !n)) err = 'Every option needs a name.'
      else if (new Set(names).size !== names.length) err = 'Option names must differ.'
      else if (names.length > 26) err = 'A question takes at most 26 options.'
    }
    if (!err && q.type === 'score') {
      const lv = q.levels.map((l) => l.trim())
      if (lv.length < 2) err = 'A score needs at least two levels.'
      else if (lv.some((l) => !l)) err = 'Every level needs a name.'
      else if (new Set(lv).size !== lv.length) err = 'Level names must differ.'
      else if (lv.length > 26) err = 'A score takes at most 26 levels.'
    }
    if (err) rows[q.key] = err
  }
  return { rows, set, ok: !set.length && !Object.keys(rows).length }
}

// ── the wire form ────────────────────────────────────────────────────────────

export interface WireQuestion {
  type: ReadType
  instructions: string
  criteria?: Record<string, string> | string[]
}

/** The `questions` map in question order (the object's key order). */
export function toWire(qs: ReadQuestion[]): Record<string, WireQuestion> {
  const out: Record<string, WireQuestion> = {}
  for (const q of qs) {
    const w: WireQuestion = { type: q.type, instructions: q.instructions.trim() }
    if (q.type === 'noul') {
      const c: Record<string, string> = {}
      if (q.yesMeans.trim()) c.true = q.yesMeans.trim()
      if (q.noMeans.trim()) c.false = q.noMeans.trim()
      if (Object.keys(c).length) w.criteria = c
    } else if (q.type === 'choice') {
      w.criteria = Object.fromEntries(q.options.map((o) => [o.name.trim(), o.description.trim()]))
    } else {
      w.criteria = q.levels.map((l) => l.trim())
    }
    out[q.id.trim()] = w
  }
  return out
}

export interface ParsedQuestions {
  questions: ReadQuestion[]
  samples?: Samples
  state?: string
  errors: string[]
}

/** The JSON tab and the import path: a `questions` map, or a whole request
 *  body holding one (`state` and `samples` come along). Rows with problems
 *  are reported and skipped; the rest load. Ids are kept as written - the
 *  runner's own rule decides them at run time, and the editor shows it. */
export function parseQuestionsJson(text: string): ParsedQuestions {
  let doc: unknown
  try {
    doc = JSON.parse(text)
  } catch (e) {
    return { questions: [], errors: [e instanceof Error ? e.message : 'not JSON'] }
  }
  return fromWire(doc)
}

export function fromWire(doc: unknown): ParsedQuestions {
  const errors: string[] = []
  if (!doc || typeof doc !== 'object' || Array.isArray(doc)) {
    return { questions: [], errors: ['The JSON must be an object mapping question ids to questions.'] }
  }
  let map = doc as Record<string, unknown>
  let samples: Samples | undefined
  let state: string | undefined
  // a whole request body: unwrap it
  if (map.questions && typeof map.questions === 'object' && !('type' in map)) {
    if (typeof map.state === 'string') state = map.state
    if (map.samples === 'auto') samples = 'auto'
    else if (typeof map.samples === 'number' && Number.isInteger(map.samples) && map.samples >= 1)
      samples = map.samples
    map = map.questions as Record<string, unknown>
  }
  const questions: ReadQuestion[] = []
  for (const [id, raw] of Object.entries(map)) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      errors.push(`${id}: a question is an object with type, instructions and criteria.`)
      continue
    }
    const w = raw as Record<string, unknown>
    const t = typeof w.type === 'string' ? w.type : ''
    const q = newQuestion()
    q.id = id
    q.idTouched = true
    q.instructions = typeof w.instructions === 'string' ? w.instructions : ''
    const c = w.criteria
    if (t === 'noul' || t === 'bool' || t === 'boolean') {
      q.type = 'noul'
      if (c && typeof c === 'object' && !Array.isArray(c)) {
        const cc = c as Record<string, unknown>
        if (typeof cc.true === 'string') q.yesMeans = cc.true
        if (typeof cc.false === 'string') q.noMeans = cc.false
      }
    } else if (t === 'choice') {
      q.type = 'choice'
      if (c && typeof c === 'object' && !Array.isArray(c)) {
        q.options = Object.entries(c as Record<string, unknown>).map(([name, d]) => ({
          name,
          description: typeof d === 'string' ? d : d == null ? '' : String(d),
        }))
      } else {
        errors.push(`${id}: choice criteria map option names to descriptions.`)
        continue
      }
    } else if (t === 'score') {
      q.type = 'score'
      if (Array.isArray(c)) {
        q.levels = c.map((l) => (typeof l === 'string' ? l : String(l)))
      } else {
        errors.push(`${id}: score criteria are an ordered list of levels.`)
        continue
      }
    } else {
      errors.push(`${id}: unknown type "${t}" (noul, choice or score).`)
      continue
    }
    questions.push(q)
  }
  const out: ParsedQuestions = { questions, errors }
  if (samples !== undefined) out.samples = samples
  if (state !== undefined) out.state = state
  return out
}

export interface ReadRequest {
  state: string
  questions: Record<string, WireQuestion>
  samples?: number
}

export function requestBody(state: string, qs: ReadQuestion[], samples: Samples): ReadRequest {
  const body: ReadRequest = { state, questions: toWire(qs) }
  if (samples !== 'auto') body.samples = samples
  return body
}

/** The equivalent curl, with the body as a heredoc - a state text has
 *  quotes and newlines, which a `-d '...'` would break on. */
export function curlFor(port: number, body: ReadRequest): string {
  return [
    `curl http://localhost:${port}/v1/systemone \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -H "Authorization: Bearer <api key>" \\`,
    `  -d @- <<'JSON'`,
    JSON.stringify(body, null, 2),
    'JSON',
  ].join('\n')
}

// ── the answer ───────────────────────────────────────────────────────────────

interface AnswerCommon {
  /** the mean probability of the reported label over the reads */
  confidence: number
  /** the share of reads that picked the reported label */
  agreement: number
  /** the mass the model put outside the label set at this slot */
  outside: number
}
export interface ReadAnswerNoul extends AnswerCommon {
  type: 'noul'
  noul: number
}
export interface ReadAnswerChoice extends AnswerCommon {
  type: 'choice'
  choice: string
  probabilities: Record<string, number>
}
export interface ReadAnswerScore extends AnswerCommon {
  type: 'score'
  score: number
  level: string
  legend: Record<string, string>
  probabilities: Record<string, number>
}
export type ReadAnswer = ReadAnswerNoul | ReadAnswerChoice | ReadAnswerScore

export interface ReadDiagRead {
  pick: string
  confidence: number
  entropy: number
}
export interface ReadDiagQuestion {
  id: string
  label: string
  position: number
  entropy: number
  label_mass: number
  /** one entry per read - a runner from before the field sends none */
  reads?: ReadDiagRead[]
}
export interface ReadResponse {
  model: string
  answers: Record<string, ReadAnswer>
  usage?: { input_tokens?: number; output_tokens?: number }
  diagnostics: {
    reads: number
    canvas: number
    questions: ReadDiagQuestion[]
    timing: { total_ms: number }
  }
}

/** The runner re-reads a slot whose entropy is above this (its `auto`
 *  threshold) - so below it the slot is settled by the runner's own
 *  measure, and above it the runner itself called it unsettled and read
 *  again. Above ln 2 the slot holds at least a coin flip between two ids.
 *  The entropy is the SLOT's over the whole vocabulary, so a slot can be
 *  unsettled while every label read agrees: the mass is then outside the
 *  labels, which the meta line prints right beside it. */
export const ENTROPY_SETTLED = 0.1
export const ENTROPY_SPLIT = Math.LN2

export type EntropyWord = 'settled' | 'unsettled' | 'split'
export function entropyWord(e: number): EntropyWord {
  if (!(e >= 0)) return 'split'
  if (e < ENTROPY_SETTLED) return 'settled'
  if (e < ENTROPY_SPLIT) return 'unsettled'
  return 'split'
}

/** Confidence in four bins with visible edges - the reference bands people
 *  already use (TypeSafe's act / review / human thresholds at 0.9 and 0.5,
 *  and 0.7 between). The number is always printed beside the colour. */
export const CONFIDENCE_EDGES = [0.5, 0.7, 0.9] as const
export const CONFIDENCE_BINS = ['under 0.5', '0.5 to 0.7', '0.7 to 0.9', '0.9 and over'] as const
export function confidenceBin(p: number): 0 | 1 | 2 | 3 {
  if (!(p >= CONFIDENCE_EDGES[0])) return 0
  if (p < CONFIDENCE_EDGES[1]) return 1
  if (p < CONFIDENCE_EDGES[2]) return 2
  return 3
}

export function fmtP(p: number | undefined): string {
  if (p === undefined || !Number.isFinite(p)) return '-'
  if (p > 0 && p < 0.005) return '<0.01'
  return p.toFixed(2)
}

export interface Bar {
  name: string
  p: number
  role: 'winner' | 'other' | 'outside'
}

const OUTSIDE = 'outside the options'

/** Ranked by mass, the winner marked, the outside-the-labels mass as its
 *  own last bar (the OpenAI `refusal` separation: never folded into a
 *  label). */
export function choiceBars(a: ReadAnswerChoice): Bar[] {
  const bars: Bar[] = Object.entries(a.probabilities)
    .map(([name, p]) => ({ name, p, role: name === a.choice ? ('winner' as const) : ('other' as const) }))
    .sort((x, y) => y.p - x.p)
  bars.push({ name: OUTSIDE, p: Math.max(0, a.outside), role: 'outside' })
  return bars
}

/** In level order - the order carries the meaning of a score. */
export function scoreBars(a: ReadAnswerScore): Bar[] {
  const bars: Bar[] = Object.keys(a.legend)
    .map(Number)
    .filter((i) => Number.isInteger(i))
    .sort((x, y) => x - y)
    .map((i) => {
      const name = a.legend[String(i)]
      return { name, p: a.probabilities[String(i)] ?? 0, role: name === a.level ? ('winner' as const) : ('other' as const) }
    })
  bars.push({ name: OUTSIDE, p: Math.max(0, a.outside), role: 'outside' })
  return bars
}

export function noulBars(a: ReadAnswerNoul): Bar[] {
  const yes = a.noul
  const bars: Bar[] = [
    { name: 'yes', p: yes, role: yes >= 0.5 ? 'winner' : 'other' },
    { name: 'no', p: 1 - yes, role: yes >= 0.5 ? 'other' : 'winner' },
  ]
  bars.push({ name: OUTSIDE, p: Math.max(0, a.outside), role: 'outside' })
  return bars
}

/** The two largest label masses within 0.05 of each other. */
export function nearTie(bars: Bar[]): boolean {
  const ps = bars
    .filter((b) => b.role !== 'outside')
    .map((b) => b.p)
    .sort((x, y) => y - x)
  return ps.length >= 2 && ps[0] - ps[1] < 0.05
}

/** Where the fractional score sits on the level scale, 0..1. */
export function scorePosition(a: ReadAnswerScore): number {
  const n = Object.keys(a.legend).length
  if (n < 2) return 0
  return Math.min(1, Math.max(0, a.score / (n - 1)))
}

/** The reported label of an answer, as the user named it. */
export function answerLabel(a: ReadAnswer): string {
  if (a.type === 'noul') return a.noul >= 0.5 ? 'yes' : 'no'
  if (a.type === 'choice') return a.choice
  return a.level
}

// ── where a runner refusal belongs on the page ──────────────────────────────

export type ErrorTarget =
  | { where: 'row'; id: string }
  | { where: 'state' }
  | { where: 'questions' }
  | { where: 'page' }

/** The runner names the question it refuses (`question "id": ...`), and its
 *  window refusal names the prompt - so each 422 lands next to the field it
 *  is about instead of in a strip at the top. */
export function routeError(msg: string): ErrorTarget {
  const m = /^question "((?:[^"\\]|\\.)+)":/.exec(msg)
  if (m) return { where: 'row', id: m[1].replace(/\\(.)/g, '$1') }
  if (/the window is|^state:/.test(msg)) return { where: 'state' }
  if (/^questions:|the answer template needs|^samples:/.test(msg)) return { where: 'questions' }
  return { where: 'page' }
}

// ── run history (local) ─────────────────────────────────────────────────────

export interface ReadRun {
  at: number
  model: string
  port: number
  /** the first line or so of the state - the run keeps no full text */
  excerpt: string
  chars: number
  questions: Record<string, WireQuestion>
  samples: Samples
  response: ReadResponse
  ms: number
}

export function excerptOf(state: string, max = 120): string {
  const s = state.trim().replace(/\s+/g, ' ')
  return s.length > max ? `${s.slice(0, max - 3)}...` : s
}
