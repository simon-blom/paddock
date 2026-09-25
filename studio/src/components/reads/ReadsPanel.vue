<script setup lang="ts">
// Reads - fixed questions asked of one text in a single diffusion pass and
// answered with probabilities: the Studio's surface for POST /v1/systemone
// on a block-diffusion model (DiffusionGemma). The body is the Jev request
// shape; the answer carries the runner's own outside-the-labels mass,
// agreement across re-reads and slot entropy on top of it. A read yields
// answers, not a reply, so this is a workbench beside Embeddings and never
// a chat lane. The loop is edit -> run -> read -> edit, so above 1200px the
// answers stay in view beside the editor. Everything runs through the same
// endpoint code calls, relayed by the manager (which holds the runner key),
// and the API pane shows the equivalent curl built from the live request.
// Earlier reads sit in a side panel the way a chat lists conversations: a
// read is the text, its questions and every run of them, kept by the manager
// and named in the URL, so a read opens again from any browser.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useModelsStore } from '@/stores/models'
import { useReadsStore } from '@/stores/reads'
import { useToastsStore } from '@/stores/toasts'
import { modelLabel } from '@/lib/model-name'
import { copyText } from '@/lib/clipboard'
import Icon from '@/components/Icon.vue'
import Collapsible from '@/components/ui/Collapsible.vue'
import Dialog from '@/components/ui/Dialog.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import MenuSeparator from '@/components/ui/MenuSeparator.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import Tabs, { type TabOption } from '@/components/ui/Tabs.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import { uuid } from '@/lib/uuid'
import { readsPreferencesApi } from '@/lib/api'
import QuestionRow from './QuestionRow.vue'
import AnswersCard from './AnswersCard.vue'
import ReadsSidebar from './ReadsSidebar.vue'
import {
  DEFAULT_MAX_QUESTIONS,
  DEFAULT_MAX_SAMPLES,
  cleanId,
  curlFor,
  deriveId,
  duplicateQuestion,
  excerptOf,
  fromWire,
  newQuestion,
  parseQuestionsJson,
  requestBody,
  routeError,
  readTitle,
  toWire,
  validate,
  withRun,
  type ReadDoc,
  type ReadQuestion,
  type ReadResponse,
  type ReadRun,
  type ReadType,
  type Samples,
} from '@/lib/reads'

const models = useModelsStore()
const sets = useReadsStore()
const toasts = useToastsStore()
const route = useRoute()
const router = useRouter()

// keep the reader list fresh while the page is open: a model started or
// stopped in the Manager appears or goes without a reload
let timer: number | undefined
async function poll(): Promise<void> {
  await models.refresh()
  await models.probeReaders()
}
onMounted(() => {
  void poll()
  void sets.refresh()
  void sets.refreshReads()
  timer = window.setInterval(() => void poll(), 5000)
})
onUnmounted(() => clearInterval(timer))

const readers = computed(() => models.readers)
const port = ref<number>(0)
watch(
  readers,
  (list) => {
    if (!list.some((m) => m.port === port.value)) port.value = list[0]?.port ?? 0
  },
  { immediate: true },
)
const current = computed(() => readers.value.find((m) => m.port === port.value))
const caps = computed(() => (current.value ? models.structuredReadFor(current.value.id) : undefined))
const maxQuestions = computed(() => caps.value?.maxQuestions ?? DEFAULT_MAX_QUESTIONS)
const modelOptions = computed<SelectOption[]>(() =>
  readers.value.map((m) => ({
    value: m.port ?? 0,
    label: m.display ?? modelLabel(m.id),
    hint: `port ${m.port}`,
    vendor: m.vendor,
    title: m.id,
  })),
)

// ── the state: the user's text, pasted or loaded from a file ───────────────
const state = ref('')
const fileName = ref('')
const extracting = ref(false)
const stateError = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
let settingFromFile = false
// typing into a loaded file's text makes it the user's text again
watch(state, () => {
  if (!settingFromFile) fileName.value = ''
})
function setFromFile(text: string, name: string): void {
  settingFromFile = true
  state.value = text
  fileName.value = name
  queueMicrotask(() => {
    settingFromFile = false
  })
}
function toB64(u8: Uint8Array): string {
  let s = ''
  const CHUNK = 0x8000
  for (let i = 0; i < u8.length; i += CHUNK) s += String.fromCharCode(...u8.subarray(i, i + CHUNK))
  return btoa(s)
}
const TEXT_EXT = /\.(txt|md|markdown|csv|tsv|json|log|xml|html?|ya?ml|toml|eml)$/i
/** Plain text is read here; anything else (PDF, docx, an image) goes
 *  through the runner's own extraction - the text the model would see. */
async function loadFile(f: File): Promise<void> {
  stateError.value = null
  if (f.type.startsWith('text/') || TEXT_EXT.test(f.name)) {
    setFromFile(await f.text(), f.name)
    return
  }
  const url = current.value ? models.extractUrl(current.value.id) : undefined
  if (!url) {
    stateError.value = 'No running server to read the file with.'
    return
  }
  extracting.value = true
  try {
    const bytes = new Uint8Array(await f.arrayBuffer())
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        filename: f.name,
        data: `data:${f.type || 'application/octet-stream'};base64,${toB64(bytes)}`,
        file_metadata: 'off',
      }),
    })
    const j = (await res.json().catch(() => null)) as { text?: string; error?: { message?: string } } | null
    if (!res.ok) {
      stateError.value = j?.error?.message ?? `extraction failed (${res.status})`
      return
    }
    setFromFile(j?.text ?? '', f.name)
  } catch (e) {
    stateError.value = e instanceof Error ? e.message : String(e)
  } finally {
    extracting.value = false
  }
}
function onDrop(e: DragEvent): void {
  const f = e.dataTransfer?.files?.[0]
  if (f) void loadFile(f)
}
/** The text goes, file or pasted - the one way back from a loaded file
 *  short of selecting it all. */
function clearState(): void {
  settingFromFile = false
  fileName.value = ''
  state.value = ''
  stateError.value = null
}
function onPick(e: Event): void {
  const el = e.target as HTMLInputElement
  const f = el.files?.[0]
  if (f) void loadFile(f)
  el.value = ''
}

// ── the questions ──────────────────────────────────────────────────────────
const questions = ref<ReadQuestion[]>([newQuestion()])
const samples = ref<Samples>('auto')
const sampleOptions = computed<SelectOption[]>(() => {
  const max = caps.value?.maxSamples ?? DEFAULT_MAX_SAMPLES
  return [
    { value: 'auto', label: 'auto', hint: 're-reads while unsettled' },
    ...[1, 2, 4, 8, 16, 32].filter((n) => n <= max).map((n) => ({ value: n, label: String(n) })),
  ]
})
function setSamples(v: string | number): void {
  samples.value = v === 'auto' ? 'auto' : Number(v)
}

function otherIds(key: string): string[] {
  return questions.value.filter((q) => q.key !== key).map((q) => q.id)
}
/** Row edits arrive as patches. The id follows the instructions until the
 *  user writes one; clearing it hands it back to the derivation. */
function onPatch(q: ReadQuestion, patch: Partial<ReadQuestion>): void {
  const { id, ...rest } = patch
  Object.assign(q, rest)
  if (id !== undefined) {
    if (!id.trim()) {
      q.idTouched = false
      q.id = deriveId(q.instructions, otherIds(q.key))
    } else {
      q.idTouched = true
      q.id = cleanId(id)
    }
  } else if (rest.instructions !== undefined && !q.idTouched) {
    q.id = deriveId(q.instructions, otherIds(q.key))
  }
  // a runner refusal pinned to this row is answered by editing it
  if (serverRowErrors.value[q.id]) {
    const { [q.id]: _, ...keep } = serverRowErrors.value
    serverRowErrors.value = keep
  }
}
function addQuestion(type: ReadType): void {
  questions.value.push(newQuestion(type))
}
function move(i: number, delta: number): void {
  const j = i + delta
  const list = questions.value
  if (j < 0 || j >= list.length) return
  ;[list[i], list[j]] = [list[j], list[i]]
}
function duplicate(i: number): void {
  const q = questions.value[i]
  questions.value.splice(
    i + 1,
    0,
    duplicateQuestion(
      q,
      questions.value.map((x) => x.id),
    ),
  )
}
function remove(i: number): void {
  questions.value.splice(i, 1)
}
// native drag between rows: the dragged row follows the pointer, so the
// order is live while dragging and settled on drop without a second step
const dragFrom = ref<number | null>(null)
function onDragEnter(i: number): void {
  const from = dragFrom.value
  if (from === null || from === i) return
  const list = questions.value
  const [item] = list.splice(from, 1)
  list.splice(i, 0, item)
  dragFrom.value = i
}

// the editor's own checks, and the runner's refusals pinned by question id
const validation = computed(() => validate(questions.value, caps.value))
const serverRowErrors = ref<Record<string, string>>({})
const questionsError = ref<string | null>(null)
const pageError = ref<string | null>(null)
/** A row's first problem - except on a row nobody has touched yet: a fresh
 *  page opening on a warning-bordered empty row is noise, and Run stays
 *  disabled through `validation.ok` either way. */
function rowError(q: ReadQuestion): string | undefined {
  const pristine = !q.instructions.trim() && !q.idTouched && !q.id
  if (pristine) return serverRowErrors.value[q.id]
  return validation.value.rows[q.key] ?? serverRowErrors.value[q.id]
}

// ── the JSON tab: the questions map as text, two-way ───────────────────────
const tab = ref<'form' | 'json'>('form')
const tabs: TabOption[] = [
  { value: 'form', label: 'Form' },
  { value: 'json', label: 'JSON' },
]
const jsonText = ref('')
const jsonError = ref<string | null>(null)
const jsonNotes = ref<string[]>([])
function setTab(v: string): void {
  if (v === 'json') {
    jsonText.value = JSON.stringify(toWire(questions.value), null, 2)
    jsonError.value = null
    jsonNotes.value = []
    tab.value = 'json'
  } else if (applyJson()) {
    tab.value = 'form'
  }
}
/** The text becomes the rows; a text that parses to nothing is refused and
 *  the tab stays, so a typo never empties the editor. */
function applyJson(): boolean {
  if (!jsonText.value.trim()) return true
  const p = parseQuestionsJson(jsonText.value)
  if (!p.questions.length && p.errors.length) {
    jsonError.value = p.errors.join(' ')
    return false
  }
  jsonError.value = null
  jsonNotes.value = p.errors
  questions.value = p.questions
  if (p.samples !== undefined) samples.value = p.samples
  if (p.state !== undefined && !state.value.trim()) state.value = p.state
  return true
}
const jsonInput = ref<HTMLInputElement | null>(null)
function onJsonPick(e: Event): void {
  const el = e.target as HTMLInputElement
  const f = el.files?.[0]
  el.value = ''
  if (!f) return
  void f.text().then((text) => {
    jsonText.value = text
    tab.value = 'json'
    applyJson()
  })
}
function exportJson(): void {
  const text = JSON.stringify({ questions: toWire(questions.value), samples: samples.value }, null, 2)
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `${(activeSet.value?.name ?? 'read-set').replace(/[^\w.-]+/g, '_')}.json`
  a.click()
  URL.revokeObjectURL(url)
}
async function copyJson(): Promise<void> {
  await copyText(jsonText.value)
  toasts.push({ tone: 'info', title: 'Copied the questions' })
}

// ── the run, and the read it belongs to ────────────────────────────────────
const busy = ref(false)
/** The read on screen, whole; null for a new read that has not run yet. */
const activeRead = ref<ReadDoc | null>(null)
const readSaveFailed = ref(false)
/** Which of its runs the answers show - the latest unless stepped back. */
const runIdx = ref(0)
const run = computed<ReadRun | null>(() => activeRead.value?.runs[runIdx.value] ?? null)
const lastMs = ref<number | null>(null)
const canRun = computed(
  () => !!current.value && !busy.value && state.value.trim().length > 0 && validation.value.ok,
)
const body = computed(() => requestBody(state.value, questions.value, samples.value))
const curl = computed(() => (current.value ? curlFor(port.value, body.value) : ''))

async function doRun(): Promise<void> {
  if (!canRun.value || !current.value) return
  busy.value = true
  pageError.value = null
  questionsError.value = null
  stateError.value = null
  serverRowErrors.value = {}
  const req = body.value
  const sourceFile = fileName.value
  const p = port.value
  const model = current.value
  const t0 = performance.now()
  try {
    const res = await fetch(`/api/runners/${p}/v1/systemone`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    })
    const json: unknown = await res.json().catch(() => null)
    const ms = Math.round(performance.now() - t0)
    if (!res.ok) {
      const msg =
        (json as { error?: { message?: string } } | null)?.error?.message ?? `HTTP ${res.status}`
      const target = routeError(msg)
      if (target.where === 'row' && questions.value.some((q) => q.id === target.id)) {
        serverRowErrors.value = { [target.id]: msg }
      } else if (target.where === 'state') {
        stateError.value = msg
      } else if (target.where === 'questions' || target.where === 'row') {
        questionsError.value = msg
      } else {
        pageError.value = msg
      }
      return
    }
    const r: ReadRun = {
      id: uuid(),
      at: Date.now(),
      model: model.display ?? modelLabel(model.id),
      port: p,
      excerpt: excerptOf(req.state),
      chars: req.state.length,
      state: req.state,
      fileName: sourceFile,
      questions: req.questions,
      samples: req.samples ?? 'auto',
      response: json as ReadResponse,
      ms,
    }
    lastMs.value = ms
    await keepRun(r)
  } catch (e) {
    pageError.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}
/** A run lands on the read on screen - a new read is created by its first
 *  run, the way a chat is created by its first message - and the read is
 *  saved whole. A failed save keeps the answers on screen and says so. */
async function keepRun(r: ReadRun): Promise<void> {
  const prev = activeRead.value
  const doc: ReadDoc = prev
    ? { ...prev, model: r.model, updatedAt: r.at, runs: withRun(prev.runs, r) }
    : {
        id: uuid(),
        title: readTitle(r.state, r.fileName),
        model: r.model,
        createdAt: r.at,
        updatedAt: r.at,
        runs: [r],
      }
  activeRead.value = doc
  runIdx.value = doc.runs.length - 1
  if (!prev) void router.replace({ name: 'reads', params: { id: doc.id } })
  await persistRead(doc)
}
async function persistRead(doc: ReadDoc): Promise<void> {
  readSaveFailed.value = true
  try {
    await sets.saveRead(doc)
    readSaveFailed.value = false
  } catch (e) {
    toasts.push({
      tone: 'bad',
      title: 'This read was not saved',
      description: e instanceof Error ? e.message : String(e),
    })
  }
}
function stepRun(delta: number): void {
  const n = activeRead.value?.runs.length ?? 0
  runIdx.value = Math.min(Math.max(runIdx.value + delta, 0), Math.max(n - 1, 0))
}
/** The answers on show belong to a request the editor has since changed. */
const stale = computed(() => {
  const r = run.value
  if (!r) return false
  const b = body.value
  return (
    b.state !== undefined &&
    ((r.state ?? '') !== b.state ||
      JSON.stringify(r.questions) !== JSON.stringify(b.questions) ||
      r.samples !== samples.value)
  )
})

// A worked example, on request only: the inputs start empty (the user's
// content, never sample prose), but a page that cannot show what a read
// looks like until one has been authored explains nothing. One click fills
// a support ticket and the three question types, then runs.
const EXAMPLE_STATE = [
  'Subject: Portal down again',
  '',
  'Hi, this is the third time this week the customer portal has gone down during business hours. We have 40 agents unable to log in right now and customers are calling. I need someone on this immediately - we are paying for the enterprise tier and this is unacceptable. Please call me back on the number on file.',
  '',
  '- Dana, Ops lead at Northwind',
].join('\n')
// The ids are DERIVED, as the editor derives them, and not short hand-picked
// words: the id is written into the answer template the model reads, and
// measured on the Q4_K_M file `urgent` / `topic` / `mood` left 0.43-0.50 of
// the mass outside the labels on every question (slot entropy 1.1-1.9)
// where `need_action_within` / `message_about` / `upset_sender` left
// 0.00-0.04 (entropy 0.05-0.22) with the same picks. A descriptive id reads
// cleaner, so the example shows the practice it recommends.
function exampleQuestions(): ReadQuestion[] {
  const urgent = newQuestion('noul')
  urgent.instructions = 'Does this message need action within the hour?'
  urgent.yesMeans = 'an outage or blocker affecting many people now'
  urgent.noMeans = 'a request that can wait a day'
  const topic = newQuestion('choice')
  topic.instructions = 'What is this message about?'
  topic.options = [
    { name: 'outage', description: 'a service is down or broken' },
    { name: 'billing', description: 'invoices, plans or payment' },
    { name: 'feature', description: 'a request for something new' },
    { name: 'other', description: 'none of these' },
  ]
  const mood = newQuestion('score')
  mood.instructions = 'How upset is the sender?'
  mood.levels = ['calm', 'annoyed', 'furious']
  const qs = [urgent, topic, mood]
  const taken: string[] = []
  for (const q of qs) {
    q.id = deriveId(q.instructions, taken)
    taken.push(q.id)
  }
  return qs
}
function loadExample(): void {
  state.value = EXAMPLE_STATE
  questions.value = exampleQuestions()
  samples.value = 'auto'
  serverRowErrors.value = {}
  questionsError.value = null
  if (tab.value === 'json') jsonText.value = JSON.stringify(toWire(questions.value), null, 2)
  void doRun()
}
function onKey(e: KeyboardEvent): void {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault()
    void doRun()
  }
}

// ── saved read sets (the Prompts pattern) ──────────────────────────────────
const activeSetId = ref<string | undefined>(undefined)
const activeSet = computed(() => sets.sets.find((s) => s.id === activeSetId.value))
const setBody = computed(() =>
  JSON.stringify({ questions: toWire(questions.value), samples: samples.value }, null, 2),
)
const dirty = computed(() => !!activeSet.value && activeSet.value.body !== setBody.value)
const setOptions = computed<SelectOption[]>(() => [
  { value: '', label: 'Unsaved set' },
  ...sets.sets.map((s) => ({ value: s.id, label: s.name })),
])
function selectSet(v: string | number): void {
  const s = sets.sets.find((x) => x.id === v)
  activeSetId.value = s?.id
  if (s) {
    try {
      const p = fromWire(JSON.parse(s.body))
      questions.value = p.questions.length ? p.questions : [newQuestion()]
      samples.value = p.samples ?? 'auto'
    } catch {
      toasts.push({ tone: 'bad', title: 'This set could not be read', description: s.name })
    }
  }
  if (tab.value === 'json') jsonText.value = JSON.stringify(toWire(questions.value), null, 2)
  serverRowErrors.value = {}
  questionsError.value = null
}
const saveOpen = ref(false)
const saveName = ref('')
const saving = ref(false)
const saveError = ref('')
const deleteOpen = ref(false)
async function saveSet(): Promise<void> {
  const s = activeSet.value
  if (!s) {
    saveAs()
    return
  }
  saving.value = true
  try {
    await sets.save(s.name, setBody.value, s.id, s.revision)
    toasts.push({ tone: 'good', title: 'Saved', description: s.name })
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Not saved', description: e instanceof Error ? e.message : String(e) })
  } finally {
    saving.value = false
  }
}
function saveAs(): void {
  saveName.value = activeSet.value ? `${activeSet.value.name} copy` : ''
  saveError.value = ''
  saveOpen.value = true
}
async function confirmSave(): Promise<void> {
  if (!saveName.value.trim() || saving.value) return
  saving.value = true
  saveError.value = ''
  try {
    const saved = await sets.save(saveName.value, setBody.value)
    activeSetId.value = saved.id
    saveOpen.value = false
    toasts.push({ tone: 'good', title: 'Saved', description: saved.name })
  } catch (e) {
    saveError.value = e instanceof Error ? e.message : String(e)
  } finally {
    saving.value = false
  }
}
async function confirmDelete(): Promise<void> {
  const s = activeSet.value
  if (!s || saving.value) return
  saving.value = true
  try {
    await sets.remove(s.id, s.revision)
    activeSetId.value = undefined
    deleteOpen.value = false
    toasts.push({ tone: 'info', title: 'Deleted', description: s.name })
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Not deleted', description: e instanceof Error ? e.message : String(e) })
  } finally {
    saving.value = false
  }
}
// ── moving between reads ───────────────────────────────────────────────────
// The URL names the read on screen (/studio/reads/<id>); no id is a new read.
// Leaving asks first only when something would be lost: text or questions
// that were never run, or edits made since the read's last run.
const hasWork = computed(
  () =>
    state.value.trim().length > 0 ||
    questions.value.some((q) => q.instructions.trim().length > 0 || q.idTouched),
)
const unsaved = computed(() => {
  if (readSaveFailed.value) return true
  if (!hasWork.value) return false
  const runs = activeRead.value?.runs ?? []
  const latest = runs[runs.length - 1]
  if (!latest) return true
  const b = body.value
  return (
    latest.state !== b.state ||
    JSON.stringify(latest.questions) !== JSON.stringify(b.questions) ||
    latest.samples !== samples.value
  )
})
/** Where a confirmed leave goes: a new read, or an earlier one. */
const leaving = ref<{ to: 'new' } | { to: 'read'; id: string } | null>(null)
function newRead(): void {
  if (unsaved.value) leaving.value = { to: 'new' }
  else goNew()
}
function openRead(id: string): void {
  if (id === activeRead.value?.id) return
  if (unsaved.value) leaving.value = { to: 'read', id }
  else void router.push({ name: 'reads', params: { id } })
}
function confirmLeave(): void {
  const l = leaving.value
  leaving.value = null
  if (!l) return
  if (l.to === 'new') goNew()
  // the route watcher opens it; the edits being left are dropped there
  else void router.push({ name: 'reads', params: { id: l.id } })
}
function goNew(): void {
  if (route.params.id) void router.push({ name: 'reads' })
  resetRead()
}
function resetRead(): void {
  clearState()
  questions.value = [newQuestion()]
  samples.value = 'auto'
  activeSetId.value = undefined
  activeRead.value = null
  readSaveFailed.value = false
  runIdx.value = 0
  lastMs.value = null
  serverRowErrors.value = {}
  questionsError.value = null
  pageError.value = null
  jsonText.value = ''
  jsonError.value = null
  jsonNotes.value = []
  tab.value = 'form'
}
/** Put a read on screen: the editor takes its latest run's text and
 *  questions, the answers show that run. */
const opening = ref(false)
async function loadRead(id: string): Promise<void> {
  opening.value = true
  pageError.value = null
  try {
    const doc = await sets.loadRead(id)
    if (route.params.id !== id) return
    resetRead()
    activeRead.value = doc
    runIdx.value = Math.max(doc.runs.length - 1, 0)
    const latest = doc.runs[doc.runs.length - 1]
    if (latest) {
      if (latest.fileName) setFromFile(latest.state ?? '', latest.fileName)
      else state.value = latest.state ?? ''
      const p = fromWire({ questions: latest.questions, samples: latest.samples })
      questions.value = p.questions.length ? p.questions : [newQuestion()]
      samples.value = p.samples ?? latest.samples
      lastMs.value = latest.ms
    }
  } catch (e) {
    if (route.params.id !== id) return
    resetRead()
    pageError.value = `This read could not be opened: ${e instanceof Error ? e.message : String(e)}`
  } finally {
    opening.value = false
  }
}
watch(
  () => route.params.id,
  (raw) => {
    const id = typeof raw === 'string' && raw ? raw : undefined
    if (!id) {
      if (activeRead.value) resetRead()
      return
    }
    if (id !== activeRead.value?.id) void loadRead(id)
  },
  { immediate: true },
)
async function renameRead(id: string, title: string): Promise<void> {
  try {
    const doc = await sets.renameRead(id, title)
    if (activeRead.value?.id === id) activeRead.value = { ...activeRead.value, title: doc.title, revision: doc.revision }
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Not renamed', description: e instanceof Error ? e.message : String(e) })
  }
}
async function removeRead(id: string): Promise<void> {
  try {
    await sets.removeRead(id)
    if (activeRead.value?.id === id) goNew()
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Not deleted', description: e instanceof Error ? e.message : String(e) })
  }
}

// the side panel folds like the chat list; a narrow window starts folded
const panelOpen = ref(window.innerWidth >= 1100)
let panelEdited = false
let restoringPanel = false
onMounted(async () => {
  try {
    const saved = await readsPreferencesApi.get()
    if (!panelEdited && typeof saved.readsPanelOpen === 'boolean') {
      restoringPanel = true
      panelOpen.value = saved.readsPanelOpen
    }
  } catch (e) {
    toasts.push({ tone: 'bad', title: 'Layout could not be loaded', description: String(e) })
  } finally { restoringPanel = false }
})
let panelSave: Promise<unknown> = Promise.resolve()
watch(panelOpen, (v) => {
  if (restoringPanel) return
  panelEdited = true
  panelSave = panelSave.catch(() => {}).then(() => readsPreferencesApi.save(v)).catch((e) => {
    toasts.push({ tone: 'bad', title: 'Layout was not saved', description: String(e) })
  })
}, { flush: 'sync' })

// ── API pane ───────────────────────────────────────────────────────────────
const copied = ref(false)
async function copyCurl(): Promise<void> {
  await copyText(curl.value)
  copied.value = true
  window.setTimeout(() => (copied.value = false), 1500)
}
</script>

<template>
  <div class="rd" @keydown="onKey">
    <ReadsSidebar
      v-if="panelOpen"
      :reads="sets.reads"
      :active-id="activeRead?.id ?? null"
      :loaded="sets.readsLoaded"
      :error="sets.readsError"
      :busy="busy"
      @new="newRead"
      @open="openRead"
      @rename="renameRead"
      @remove="removeRead"
      @fold="panelOpen = false"
    />
    <aside v-else class="rd__rail">
      <Tooltip label="Show reads" side="right">
        <button class="pk-icon-btn rd__railbtn" type="button" aria-label="Show reads" @click="panelOpen = true">
          <Icon name="panel-left" :size="18" />
        </button>
      </Tooltip>
      <Tooltip label="New read" side="right">
        <button class="pk-icon-btn rd__railbtn" type="button" aria-label="New read" :disabled="busy" @click="newRead">
          <Icon name="plus" :size="18" />
        </button>
      </Tooltip>
    </aside>

    <div class="rd__main">
    <div class="rd__inner">
    <div class="rd__head">
      <div>
        <h1 class="rd__title">Reads</h1>
        <p class="rd__lead">
          Ask fixed questions about a text and get probabilities back - the same /v1/systemone your
          code calls.
        </p>
      </div>
      <div v-if="readers.length" class="rd__headr">
        <Select v-model="port" :options="modelOptions" />
      </div>
    </div>

    <div v-if="opening && !activeRead" class="rd__none">
      <Icon name="spinner" :size="24" class="rd__none-icon spin" />
      <p class="rd__none-txt">Opening the read...</p>
    </div>
    <div v-else-if="!readers.length && !activeRead && !models.readersProbed" class="rd__none">
      <Icon name="spinner" :size="24" class="rd__none-icon spin" />
      <p class="rd__none-txt">Looking for a running model that reads...</p>
    </div>
    <div v-else-if="!readers.length && !activeRead" class="rd__none">
      <Icon name="list-checks" :size="32" class="rd__none-icon" />
      <p class="rd__none-title">No model that can read is running</p>
      <p class="rd__none-txt">
        Reads need a block-diffusion model - start DiffusionGemma in the Manager.
      </p>
      <RouterLink class="pk-btn pk-btn--primary" :to="{ name: 'server-new' }">
        <Icon name="play" :size="14" /> Start a model
      </RouterLink>
    </div>

    <template v-else>
      <div v-if="pageError" class="rd__error" role="alert">{{ pageError }}</div>
      <p v-if="!readers.length" class="rd__noreader">
        No model that can read is running, so this read cannot run again until DiffusionGemma is
        started in the Manager.
      </p>

      <div class="rd__cols">
        <div class="rd__left">
          <section class="rd__card">
            <div class="rd__cardhead">
              <h2 class="rd__h2">State</h2>
              <span class="rd__count">{{ state.length }} characters</span>
            </div>
            <textarea
              v-model="state"
              class="pk-input rd__ta"
              rows="9"
              spellcheck="false"
              placeholder="Paste the text to read - an email, a ticket, a transcript, a document."
              @dragover.prevent
              @drop.prevent="onDrop"
            />
            <div class="rd__filerow">
              <button
                class="pk-btn pk-btn--sm"
                type="button"
                :disabled="extracting"
                @click="fileInput?.click()"
              >
                <Icon :name="extracting ? 'spinner' : 'upload'" :size="13" :class="{ spin: extracting }" />
                {{ extracting ? 'Reading the file...' : 'Load a file' }}
              </button>
              <input ref="fileInput" type="file" class="rd__hidden" @change="onPick" />
              <span v-if="fileName" class="rd__chip">
                <Icon name="file" :size="12" />
                <span class="rd__chipname">{{ fileName }}</span>
                <button class="rd__chipx" type="button" aria-label="Remove the file" @click="clearState">
                  <Icon name="x" :size="11" />
                </button>
              </span>
              <button
                v-else-if="state.length"
                class="pk-btn pk-btn--sm pk-btn--ghost"
                type="button"
                @click="clearState"
              >
                <Icon name="x" :size="13" /> Clear
              </button>
              <span v-else class="rd__hintline">or drop a file on the text</span>
            </div>
            <p v-if="stateError" class="rd__hint rd__hint--warn" role="alert">{{ stateError }}</p>
          </section>

          <section class="rd__card">
            <div class="rd__cardhead">
              <h2 class="rd__h2">Questions</h2>
              <span class="rd__count">{{ questions.length }} of {{ maxQuestions }}</span>
              <div class="rd__sets">
                <button
                  v-if="validation.ok && (dirty || !activeSet)"
                  class="pk-btn pk-btn--sm pk-btn--ghost"
                  type="button"
                  :disabled="saving"
                  @click="saveSet"
                >
                  <Icon name="save" :size="13" /> {{ activeSet ? 'Save' : 'Save set' }}
                </button>
                <Select
                  :model-value="activeSetId ?? ''"
                  :options="setOptions"
                  @update:model-value="selectSet"
                />
                <Menu>
                  <MenuTrigger>
                    <button class="pk-icon-btn" type="button" aria-label="Set actions">
                      <Icon name="more-horizontal" :size="16" />
                    </button>
                  </MenuTrigger>
                  <MenuContent align="end" label="Set actions">
                    <MenuItem :disabled="saving" @select="saveSet">
                      <Icon name="save" :size="14" /> {{ activeSet ? 'Save' : 'Save as...' }}
                    </MenuItem>
                    <MenuItem v-if="activeSet" :disabled="saving" @select="saveAs">
                      <Icon name="copy" :size="14" /> Save as...
                    </MenuItem>
                    <MenuSeparator />
                    <MenuItem @select="jsonInput?.click()">
                      <Icon name="upload" :size="14" /> Import JSON...
                    </MenuItem>
                    <MenuItem @select="exportJson"><Icon name="download" :size="14" /> Export JSON</MenuItem>
                    <template v-if="activeSet">
                      <MenuSeparator />
                      <MenuItem danger @select="deleteOpen = true">
                        <Icon name="trash" :size="14" /> Delete set
                      </MenuItem>
                    </template>
                  </MenuContent>
                </Menu>
              </div>
            </div>

            <div class="rd__qbar">
              <Tabs :model-value="tab" :tabs="tabs" @update:model-value="setTab" />
              <label class="rd__samples">
                <span>Reads per question</span>
                <Select :model-value="samples" :options="sampleOptions" @update:model-value="setSamples" />
              </label>
            </div>

            <p v-if="questionsError" class="rd__hint rd__hint--warn" role="alert">{{ questionsError }}</p>
            <p
              v-for="m in validation.set"
              :key="m"
              class="rd__hint"
              :class="{ 'rd__hint--warn': questions.length > 0 }"
            >
              {{ m }}
            </p>

            <template v-if="tab === 'form'">
              <div class="rd__rows">
                <QuestionRow
                  v-for="(q, i) in questions"
                  :key="q.key"
                  :q="q"
                  :index="i"
                  :count="questions.length"
                  :error="rowError(q)"
                  :types="caps?.types ?? []"
                  :dragging="dragFrom === i"
                  :answer="run?.response.answers[q.id]"
                  @patch="onPatch(q, $event)"
                  @move="move(i, $event)"
                  @duplicate="duplicate(i)"
                  @remove="remove(i)"
                  @dragstart="dragFrom = i"
                  @dragenter="onDragEnter(i)"
                  @dragend="dragFrom = null"
                />
              </div>
              <div class="rd__addrow">
                <span class="rd__addlabel">Add</span>
                <button
                  class="pk-btn pk-btn--sm"
                  type="button"
                  :disabled="questions.length >= maxQuestions"
                  @click="addQuestion('noul')"
                >
                  <Icon name="plus" :size="13" /> Yes / no
                </button>
                <button
                  class="pk-btn pk-btn--sm"
                  type="button"
                  :disabled="questions.length >= maxQuestions"
                  @click="addQuestion('choice')"
                >
                  <Icon name="plus" :size="13" /> Choice
                </button>
                <button
                  class="pk-btn pk-btn--sm"
                  type="button"
                  :disabled="questions.length >= maxQuestions"
                  @click="addQuestion('score')"
                >
                  <Icon name="plus" :size="13" /> Score
                </button>
              </div>
            </template>

            <template v-else>
              <textarea
                v-model="jsonText"
                class="pk-input rd__json"
                rows="16"
                spellcheck="false"
                @blur="applyJson"
              />
              <p v-if="jsonError" class="rd__hint rd__hint--warn" role="alert">{{ jsonError }}</p>
              <p v-for="n in jsonNotes" :key="n" class="rd__hint rd__hint--warn">{{ n }}</p>
              <div class="rd__addrow">
                <button class="pk-btn pk-btn--sm" type="button" @click="applyJson">Apply</button>
                <button class="pk-btn pk-btn--sm pk-btn--ghost" type="button" @click="copyJson">
                  <Icon name="copy" :size="13" /> Copy
                </button>
              </div>
            </template>

            <div class="rd__actions">
              <button class="pk-btn pk-btn--primary" type="button" :disabled="!canRun" @click="doRun">
                <Icon :name="busy ? 'spinner' : 'play'" :size="14" :class="{ spin: busy }" />
                Run
              </button>
              <span class="rd__runhint">Ctrl+Enter</span>
              <span v-if="lastMs !== null" class="rd__meta">{{ lastMs }} ms</span>
            </div>
            <input
              ref="jsonInput"
              type="file"
              accept=".json,application/json"
              class="rd__hidden"
              @change="onJsonPick"
            />
          </section>

          <Collapsible class="rd__api" summary="API call" hint="the body is the /v1/systemone shape">
            <div class="rd__apirow">
              <button class="pk-btn pk-btn--sm pk-btn--ghost" type="button" @click="copyCurl">
                <Icon :name="copied ? 'check' : 'copy'" :size="13" /> {{ copied ? 'Copied' : 'Copy' }}
              </button>
            </div>
            <pre class="rd__pre">{{ curl }}</pre>
          </Collapsible>
        </div>

        <div class="rd__right">
          <div class="rd__card rd__card--answers">
            <button v-if="readSaveFailed && activeRead" type="button" class="pk-btn pk-btn--sm" :disabled="busy" @click="persistRead(activeRead)">Retry saving</button>
            <p v-if="run?.stateMissing" class="rd__meta">The original input was not retained with this older result.</p>
            <AnswersCard
              :run="run"
              :run-index="runIdx"
              :run-count="activeRead?.runs.length ?? 0"
              :busy="busy"
              :stale="stale"
              :can-example="!!current && !busy"
              @step="stepRun"
              @example="loadExample"
            />
          </div>
        </div>
      </div>
    </template>
    </div>
    </div>

    <Dialog :open="saveOpen" title="Save read set" icon="save" size="sm" :busy="saving" @close="saveOpen = false">
      <label class="rd__field">
        <span class="rd__label">Name</span>
        <input
          v-model="saveName"
          class="pk-input"
          placeholder="Ticket triage"
          @keydown.enter.prevent="confirmSave"
        />
      </label>
      <p v-if="saveError" class="rd__hint rd__hint--warn" role="alert">{{ saveError }}</p>
      <template #footer>
        <button class="pk-btn" type="button" :disabled="saving" @click="saveOpen = false">Cancel</button>
        <button
          class="pk-btn pk-btn--primary"
          type="button"
          :disabled="!saveName.trim() || saving"
          @click="confirmSave"
        >
          Save
        </button>
      </template>
    </Dialog>

    <Dialog
      :open="!!leaving"
      title="Leave this read?"
      icon="alert-triangle"
      role="alertdialog"
      size="sm"
      @close="leaving = null"
    >
      <p class="rd__dlgtext">
        {{
          activeRead
            ? 'The edits made since this read last ran are not kept. Run it to keep them.'
            : 'This text and these questions have not been run, so they are not kept yet.'
        }}
      </p>
      <template #footer>
        <button class="pk-btn" type="button" @click="leaving = null">Cancel</button>
        <button class="pk-btn pk-btn--primary" type="button" @click="confirmLeave">Leave without them</button>
      </template>
    </Dialog>

    <Dialog
      :open="deleteOpen"
      title="Delete this read set?"
      icon="trash"
      danger
      role="alertdialog"
      size="sm"
      :busy="saving"
      @close="deleteOpen = false"
    >
      <p class="rd__dlgtext">
        "{{ activeSet?.name }}" is removed. The questions stay in the editor, and reads made with it
        stay in the list.
      </p>
      <template #footer>
        <button class="pk-btn" type="button" :disabled="saving" @click="deleteOpen = false">Cancel</button>
        <button class="pk-btn pk-btn--danger" type="button" :disabled="saving" @click="confirmDelete">
          Delete
        </button>
      </template>
    </Dialog>
  </div>
</template>

<style scoped>
/* two panes, the chat shape: the history on the left at full height, the
   workspace scrolling on its own beside it */
.rd {
  display: flex;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.rd__main {
  flex: 1;
  min-width: 0;
  overflow: auto;
  padding: 32px;
}
.rd__inner {
  max-width: var(--pk-panel-width);
  margin: 0 auto;
}
.rd__rail {
  flex: none;
  width: 48px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 12px 0;
  background: var(--pk-bg-surface);
  border-right: 1px solid var(--pk-border-default);
}
.rd__railbtn {
  width: 34px;
  height: 34px;
}
.rd__noreader {
  margin: 0 0 12px;
  padding: 8px 12px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-surface);
  border: 1px solid var(--pk-border-default);
  color: var(--pk-text-secondary);
  font-size: var(--pk-font-size-sm);
}
.rd__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}
.rd__title {
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--pk-text-primary);
  margin-bottom: 4px;
}
.rd__lead {
  margin: 0;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.rd__headr {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: none;
}
.rd__none {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 64px 24px;
  text-align: center;
}
.rd__none-icon {
  color: var(--pk-text-muted);
}
.rd__none-title {
  margin: 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--pk-text-primary);
}
.rd__none-txt {
  margin: 0 0 6px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.rd__error {
  color: var(--pk-text-danger);
  background: var(--pk-bg-danger-subtle);
  border-radius: var(--pk-radius-md);
  padding: 10px 14px;
  margin-bottom: 12px;
  font-size: var(--pk-font-size-sm);
}
.rd__cols {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.rd__left {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
}
.rd__right {
  min-width: 0;
}
/* two columns once both fit: the answers stay in view while the questions
   are edited, which is the whole loop */
@media (min-width: 1200px) {
  .rd__inner {
    max-width: 1400px;
  }
  .rd__cols {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    align-items: start;
  }
  .rd__right {
    position: sticky;
    top: 0;
  }
  .rd__card--answers {
    max-height: calc(100vh - var(--pk-header-height) - 64px);
    overflow: auto;
  }
}
/* the surface card: fields sit on bg-surface, rows step down to bg-base */
.rd__card {
  display: flex;
  flex-direction: column;
  gap: 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-surface);
  padding: 16px 20px 20px;
}
.rd__cardhead {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.rd__h2 {
  margin: 0;
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.rd__count {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.rd__sets {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
}
.rd__ta {
  height: auto;
  padding: 10px 12px;
  resize: vertical;
  line-height: 1.55;
  font-family: inherit;
  background: var(--pk-bg-inset);
}
.rd__json {
  height: auto;
  padding: 10px 12px;
  resize: vertical;
  line-height: 1.5;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  background: var(--pk-bg-inset);
}
.rd__filerow {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.rd__hidden {
  display: none;
}
.rd__chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  max-width: 320px;
  padding: 2px 4px 2px 10px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
  border: 1px solid var(--pk-border-default);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
.rd__chipname {
  min-width: 0;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.rd__chipx {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: 0;
  border-radius: var(--pk-radius-full);
  background: transparent;
  color: var(--pk-text-muted);
  cursor: pointer;
}
.rd__chipx:hover {
  color: var(--pk-text-primary);
  background: var(--pk-bg-hover);
}
.rd__hintline,
.rd__hint {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.rd__hint--warn {
  color: var(--pk-status-warning);
}
.rd__qbar {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.rd__samples {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding-bottom: 4px;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
.rd__rows {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.rd__addrow {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}
.rd__addlabel {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  margin-right: 2px;
}
/* Run stays in reach: the bar sticks to the bottom of the scrollport while
   the Questions card is in view, on the card's own surface, spanning the
   card's padding so it reads as the card's edge rather than a floating
   strip */
.rd__actions {
  position: sticky;
  bottom: 0;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 -20px -20px;
  padding: 10px 20px 14px;
  border-top: 1px solid var(--pk-border-default);
  border-radius: 0 0 var(--pk-radius-lg) var(--pk-radius-lg);
  background: var(--pk-bg-surface);
}
.rd__runhint {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.rd__meta {
  margin-left: auto;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.rd__api {
  margin-top: -4px;
}
.rd__apirow {
  display: flex;
  justify-content: flex-end;
}
.rd__pre {
  margin: 4px 0 0;
  padding: 10px 12px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-size: var(--pk-font-size-xs);
  overflow-x: auto;
  max-height: 360px;
}
.rd__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.rd__label {
  font-size: var(--pk-font-size-sm);
  font-weight: 500;
  color: var(--pk-text-secondary);
}
.rd__dlgtext {
  margin: 0;
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  line-height: 1.5;
}
.spin {
  animation: pk-spin 0.8s linear infinite;
}
@keyframes pk-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
