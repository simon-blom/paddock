<script setup lang="ts">
// The answers of one read, one block per question in the run's own order:
// the reported label first, the uncertainty beside it and never inside the
// control (an uncertainty painted into the answer anchors the reader). A
// yes/no is a marker on a 0-1 rail with faint reference marks; a choice is
// ranked bars with the outside-the-options mass as its own bar; a score is
// the marker on its level scale over the per-level bars. Colour is a binned
// confidence swatch with a legend, and the number is always printed.
import { computed } from 'vue'
import Collapsible from '@/components/ui/Collapsible.vue'
import Icon from '@/components/Icon.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import ProbBars from './ProbBars.vue'
import ReadHeatmap from './ReadHeatmap.vue'
import { fmtClock } from '@/lib/format'
import {
  CONFIDENCE_BINS,
  answerLabel,
  choiceBars,
  confidenceBin,
  entropyWord,
  fmtP,
  nearTie,
  scoreBars,
  scorePosition,
  type Bar,
  type ReadAnswer,
  type ReadAnswerScore,
  type ReadRun,
} from '@/lib/reads'

const props = defineProps<{
  run: ReadRun | null
  history: ReadRun[]
  /** index into `history` of the run shown, when it is one of them */
  selected: number
  busy: boolean
  /** the editor has moved on since this read: the answers on show are of
   *  a request that is no longer the one Run would send */
  stale: boolean
  /** a reader is up and the page can fill and run its worked example */
  canExample: boolean
}>()
const emit = defineEmits<{ (e: 'select', index: number): void; (e: 'example'): void }>()

const historyOptions = computed<SelectOption[]>(() =>
  props.history.map((r, i) => ({
    value: i,
    label: `${fmtClock(new Date(r.at))} ${r.excerpt}`.slice(0, 72),
    hint: `${r.response.diagnostics.reads} read${r.response.diagnostics.reads === 1 ? '' : 's'}`,
  })),
)

interface Block {
  id: string
  instructions: string
  answer: ReadAnswer
  label: string
  bars: Bar[] | null
  tie: boolean
  entropy: number
  reads: number
  agreement: string | null
}

const blocks = computed<Block[]>(() => {
  const run = props.run
  if (!run) return []
  const reads = run.response.diagnostics.reads
  const diag = new Map(run.response.diagnostics.questions.map((d) => [d.id, d]))
  // the run's own question order; answers the runner added on its own (none
  // today) trail behind
  const ids = Object.keys(run.questions)
  for (const id of Object.keys(run.response.answers)) if (!ids.includes(id)) ids.push(id)
  return ids.flatMap((id) => {
    const answer = run.response.answers[id]
    if (!answer) return []
    const bars =
      answer.type === 'choice' ? choiceBars(answer) : answer.type === 'score' ? scoreBars(answer) : null
    const k = Math.round(answer.agreement * reads)
    return [
      {
        id,
        instructions: run.questions[id]?.instructions ?? '',
        answer,
        label: answerLabel(answer),
        bars,
        tie: bars ? nearTie(bars) : Math.abs((answer.type === 'noul' ? answer.noul : 0.5) - 0.5) < 0.025,
        entropy: diag.get(id)?.entropy ?? Number.NaN,
        reads,
        agreement: reads > 1 ? `${k} of ${reads} reads agree` : null,
      },
    ]
  })
})

const heatRows = computed(() => {
  const run = props.run
  if (!run || run.response.diagnostics.reads < 2) return []
  const rows = run.response.diagnostics.questions
    .filter((d) => d.reads && d.reads.length > 1)
    .map((d) => ({ id: d.id, reads: d.reads ?? [] }))
  return rows
})

function scaleLabels(a: ReadAnswerScore): { name: string; at: number; show: boolean }[] {
  const names = Object.keys(a.legend)
    .map(Number)
    .sort((x, y) => x - y)
    .map((i) => a.legend[String(i)])
  const n = names.length
  return names.map((name, i) => ({
    name,
    at: n > 1 ? i / (n - 1) : 0,
    // six or more levels cannot all be lettered on one line: keep the ends
    // and the reported one
    show: n <= 5 || i === 0 || i === n - 1 || name === a.level,
  }))
}

const raw = computed(() => (props.run ? JSON.stringify(props.run.response, null, 2) : ''))
</script>

<template>
  <section class="ac">
    <div class="ac__head">
      <h2 class="ac__title">Answers</h2>
      <span v-if="busy" class="ac__meta ac__meta--live">
        <Icon name="spinner" :size="12" class="spin" /> reading
      </span>
      <span v-else-if="run" class="ac__meta">
        {{ run.ms }} ms · {{ run.response.diagnostics.reads }}
        read{{ run.response.diagnostics.reads === 1 ? '' : 's' }}
      </span>
      <span v-if="run && stale && !busy" class="ac__stale">edited since this read</span>
      <Select
        v-if="history.length > 1"
        class="ac__hist"
        :model-value="selected"
        :options="historyOptions"
        @update:model-value="(v) => emit('select', Number(v))"
      />
    </div>

    <div v-if="busy && !run" class="ac__empty">
      <Icon name="spinner" :size="20" class="spin" />
      <p>Reading...</p>
    </div>
    <div v-else-if="!run" class="ac__empty">
      <p>Run a read to see the answers here.</p>
      <button v-if="canExample" class="pk-btn pk-btn--sm" type="button" @click="emit('example')">
        <Icon name="play" :size="13" /> Try an example
      </button>
    </div>

    <template v-else>
      <article v-for="b in blocks" :key="b.id" class="ac__q">
        <header class="ac__qhead">
          <span class="ac__qid">{{ b.id }}</span>
          <span class="ac__qtext">{{ b.instructions }}</span>
        </header>

        <div class="ac__answer">
          <span class="ac__label">{{ b.label }}</span>
          <span class="ac__conf">
            <i class="ac__dot" :class="`ac__dot--${confidenceBin(b.answer.confidence)}`" />
            {{ fmtP(b.answer.confidence) }}
          </span>
          <span v-if="b.tie" class="ac__tie">near tie</span>
        </div>

        <div v-if="b.answer.type === 'noul'" class="rail">
          <div class="rail__track">
            <span class="rail__tick" :style="{ left: '30%' }" />
            <span class="rail__tick" :style="{ left: '70%' }" />
            <span class="rail__fill" :style="{ width: `${b.answer.noul * 100}%` }" />
            <span class="rail__marker" :style="{ left: `${b.answer.noul * 100}%` }" />
          </div>
          <div class="rail__scale">
            <span>no</span>
            <span class="rail__value">p(yes) {{ fmtP(b.answer.noul) }}</span>
            <span>yes</span>
          </div>
        </div>

        <div v-else-if="b.answer.type === 'score'" class="scale">
          <div class="scale__track">
            <span
              v-for="l in scaleLabels(b.answer)"
              :key="l.name"
              class="scale__stop"
              :style="{ left: `${l.at * 100}%` }"
            />
            <span class="scale__marker" :style="{ left: `${scorePosition(b.answer) * 100}%` }" />
          </div>
          <div class="scale__labels">
            <span
              v-for="l in scaleLabels(b.answer)"
              :key="l.name"
              class="scale__label"
              :class="{ 'scale__label--on': l.name === b.answer.level, 'scale__label--hide': !l.show }"
              :style="{ left: `${l.at * 100}%` }"
            >
              {{ l.name }}
            </span>
          </div>
          <p class="ac__score">score {{ b.answer.score.toFixed(2) }} on 0 to {{ Object.keys(b.answer.legend).length - 1 }}</p>
          <ProbBars v-if="b.bars" :bars="b.bars" />
        </div>

        <ProbBars v-else-if="b.bars" :bars="b.bars" />

        <p class="ac__line">
          <template v-if="b.agreement">{{ b.agreement }} · </template>
          <template v-if="Number.isFinite(b.entropy)">slot entropy {{ b.entropy.toFixed(2) }} ({{ entropyWord(b.entropy) }})</template>
          <template v-else>slot entropy unknown</template>
          · outside {{ fmtP(b.answer.outside) }}
        </p>
      </article>

      <div class="ac__legend">
        <span class="ac__legend-h">confidence</span>
        <span v-for="(name, i) in CONFIDENCE_BINS" :key="name" class="ac__legend-i">
          <i class="ac__dot" :class="`ac__dot--${i}`" /> {{ name }}
        </span>
      </div>

      <Collapsible class="ac__diag" summary="Diagnostics" :hint="run.model">
        <table class="ac__table">
          <tbody>
            <tr>
              <th>canvas</th>
              <td class="c-num">{{ run.response.diagnostics.canvas }} positions</td>
              <th>reads</th>
              <td class="c-num">{{ run.response.diagnostics.reads }}</td>
              <th>prompt</th>
              <td class="c-num">{{ run.response.usage?.input_tokens ?? '-' }} tokens</td>
              <th>time</th>
              <td class="c-num">{{ Math.round(run.response.diagnostics.timing.total_ms) }} ms</td>
            </tr>
          </tbody>
        </table>
        <table class="ac__table ac__table--q">
          <thead>
            <tr>
              <th>question</th>
              <th>label</th>
              <th class="c-num">position</th>
              <th class="c-num">entropy</th>
              <th class="c-num">label mass</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in run.response.diagnostics.questions" :key="d.id">
              <td class="ac__mono">{{ d.id }}</td>
              <td class="ac__mono">{{ d.label }}</td>
              <td class="c-num">{{ d.position }}</td>
              <td class="c-num">{{ Number.isFinite(d.entropy) ? d.entropy.toFixed(3) : '-' }}</td>
              <td class="c-num">{{ fmtP(d.label_mass) }}</td>
            </tr>
          </tbody>
        </table>
        <ReadHeatmap v-if="heatRows.length" :rows="heatRows" />
        <pre class="ac__raw">{{ raw }}</pre>
      </Collapsible>
    </template>
  </section>
</template>

<style scoped>
.ac {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.ac__head {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.ac__title {
  margin: 0;
  font-size: var(--pk-font-size-base);
  font-weight: 600;
  color: var(--pk-text-primary);
}
.ac__meta {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.ac__meta--live {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--pk-text-secondary);
}
.ac__stale {
  padding: 1px 8px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-xs);
}
.ac__hist {
  margin-left: auto;
}
.ac__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 40px 16px;
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.ac__empty p {
  margin: 0;
}
.ac__q {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px 0 14px;
  border-top: 1px solid var(--pk-border-default);
}
.ac__qhead {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
}
.ac__qid {
  flex: none;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-accent-text);
}
.ac__qtext {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  overflow-wrap: anywhere;
}
.ac__answer {
  display: flex;
  align-items: baseline;
  gap: 12px;
}
.ac__label {
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--pk-text-primary);
  overflow-wrap: anywhere;
}
.ac__conf {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}
.ac__dot {
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  border: 1px solid var(--pk-border-subtle);
}
.ac__dot--0 {
  background: var(--pk-conf-1);
}
.ac__dot--1 {
  background: var(--pk-conf-2);
}
.ac__dot--2 {
  background: var(--pk-conf-3);
}
.ac__dot--3 {
  background: var(--pk-conf-4);
}
.ac__tie {
  padding: 1px 7px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-status-warning-subtle);
  color: var(--pk-status-warning);
  font-size: var(--pk-font-size-xs);
}
.ac__line {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  font-variant-numeric: tabular-nums;
}
.ac__score {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
  font-variant-numeric: tabular-nums;
}

/* the yes/no rail: a line from no to yes, the marker at p(yes), two faint
   marks at 0.3 and 0.7 - reference points, not a verdict */
.rail {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 6px 0 2px;
}
.rail__track {
  position: relative;
  height: 6px;
  margin: 0 5px;
  border-radius: 3px;
  background: var(--pk-border-default);
}
.rail__fill {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  border-radius: 3px;
  background: var(--pk-accent-subtle);
}
.rail__tick {
  position: absolute;
  top: -4px;
  bottom: -4px;
  width: 1px;
  background: var(--pk-border-strong);
  opacity: 0.6;
}
.rail__marker {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--pk-accent);
  border: 2px solid var(--pk-bg-surface);
  transform: translate(-50%, -50%);
  box-shadow: 0 0 0 1px var(--pk-accent);
}
.rail__scale {
  display: flex;
  justify-content: space-between;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.rail__value {
  font-family: var(--pk-font-mono);
  color: var(--pk-text-secondary);
}

/* the score scale: one stop per level, the marker at the fractional score */
.scale {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-top: 6px;
}
.scale__track {
  position: relative;
  height: 6px;
  margin: 0 5px;
  border-radius: 3px;
  background: var(--pk-border-default);
}
.scale__stop {
  position: absolute;
  top: 50%;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--pk-border-strong);
  transform: translate(-50%, -50%);
}
.scale__marker {
  position: absolute;
  top: 50%;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--pk-accent);
  border: 2px solid var(--pk-bg-surface);
  transform: translate(-50%, -50%);
  box-shadow: 0 0 0 1px var(--pk-accent);
}
.scale__labels {
  position: relative;
  height: 18px;
  margin: 0 5px 4px;
}
.scale__label {
  position: absolute;
  top: 0;
  transform: translateX(-50%);
  max-width: 34%;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.scale__label:first-child {
  transform: none;
}
.scale__label:last-child {
  transform: translateX(-100%);
}
.scale__label--on {
  color: var(--pk-text-primary);
  font-weight: 600;
}
.scale__label--hide {
  visibility: hidden;
}

.ac__legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 14px;
  padding-top: 10px;
  border-top: 1px solid var(--pk-border-default);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.ac__legend-h {
  color: var(--pk-text-secondary);
}
.ac__legend-i {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.ac__diag :deep(.pk-coll__body) {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-left: 0;
}
.ac__table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--pk-font-size-xs);
}
.ac__table th {
  text-align: left;
  font-weight: 500;
  color: var(--pk-text-muted);
  padding: 4px 10px 4px 0;
  white-space: nowrap;
}
.ac__table td {
  padding: 4px 14px 4px 0;
  color: var(--pk-text-primary);
}
.ac__table--q th,
.ac__table--q td {
  border-bottom: 1px solid var(--pk-border-subtle);
}
.ac__table--q thead th {
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 0.72rem;
}
.ac__table th.c-num,
.ac__table td.c-num {
  text-align: right;
}
.ac__mono {
  font-family: var(--pk-font-mono);
}
.c-num {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.ac__raw {
  margin: 0;
  max-height: 320px;
  padding: 10px 12px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-size: var(--pk-font-size-xs);
  overflow: auto;
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
