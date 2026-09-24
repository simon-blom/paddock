<script setup lang="ts">
// One question in the Reads editor: its type, its id, the instructions and
// the per-type criteria, with the row adorners a survey builder has (grab
// handle, move, duplicate, delete). Edits go up as patches; the page owns
// the rows, derives ids and validates. The row never mutates its prop.
//
// Every type's criteria are kept on the question at once, so the type
// picker is lossless: switch to score and back and the options are still
// there.
import { ref, computed } from 'vue'
import Icon from '@/components/Icon.vue'
import Menu from '@/components/ui/Menu.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import MenuItem from '@/components/ui/MenuItem.vue'
import MenuSeparator from '@/components/ui/MenuSeparator.vue'
import MenuTrigger from '@/components/ui/MenuTrigger.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import Tooltip from '@/components/ui/Tooltip.vue'
import {
  READ_TYPES,
  answerLabel,
  confidenceBin,
  fmtP,
  type ReadAnswer,
  type ReadOption,
  type ReadQuestion,
  type ReadType,
} from '@/lib/reads'

const props = defineProps<{
  q: ReadQuestion
  index: number
  count: number
  /** the first problem with this row - the editor's own check or the
   *  runner's refusal naming this question */
  error?: string | undefined
  /** the types the running model reads (its `structured_read.types`) */
  types: string[]
  dragging: boolean
  /** this question's answer in the read on show, so the row says what it
   *  got without a trip to the other column */
  answer?: ReadAnswer | undefined
}>()
const emit = defineEmits<{
  (e: 'patch', patch: Partial<ReadQuestion>): void
  (e: 'move', delta: number): void
  (e: 'duplicate'): void
  (e: 'remove'): void
  (e: 'dragstart'): void
  (e: 'dragenter'): void
  (e: 'dragend'): void
}>()

const typeOptions = computed<SelectOption[]>(() =>
  READ_TYPES.map((t) => ({
    value: t.value,
    label: t.label,
    disabled: props.types.length > 0 && !props.types.includes(t.value),
  })),
)

function setType(v: string | number): void {
  emit('patch', { type: v as ReadType })
}
function input(e: Event): string {
  return (e.target as HTMLInputElement | HTMLTextAreaElement).value
}

// choice options
function setOption(i: number, field: keyof ReadOption, e: Event): void {
  const options = props.q.options.map((o, j) => (j === i ? { ...o, [field]: input(e) } : o))
  emit('patch', { options })
}
function addOption(): void {
  emit('patch', { options: [...props.q.options, { name: '', description: '' }] })
}
function removeOption(i: number): void {
  emit('patch', { options: props.q.options.filter((_, j) => j !== i) })
}

// score levels (ordered: the order is the scale)
function setLevel(i: number, e: Event): void {
  emit('patch', { levels: props.q.levels.map((l, j) => (j === i ? input(e) : l)) })
}
function addLevel(): void {
  emit('patch', { levels: [...props.q.levels, ''] })
}
function removeLevel(i: number): void {
  emit('patch', { levels: props.q.levels.filter((_, j) => j !== i) })
}
function moveLevel(i: number, delta: number): void {
  const j = i + delta
  if (j < 0 || j >= props.q.levels.length) return
  const levels = [...props.q.levels]
  ;[levels[i], levels[j]] = [levels[j], levels[i]]
  emit('patch', { levels })
}

// Native drag is armed from the grab handle only: a row that is always
// draggable steals text selection inside its own inputs.
const grab = ref(false)
function onDragStart(e: DragEvent): void {
  if (!grab.value) {
    e.preventDefault()
    return
  }
  e.dataTransfer?.setData('text/plain', props.q.id)
  if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move'
  emit('dragstart')
}
function onDragEnd(): void {
  grab.value = false
  emit('dragend')
}
</script>

<template>
  <div
    class="qr"
    :class="{ 'qr--bad': !!error, 'qr--drag': dragging }"
    :draggable="grab"
    @dragstart="onDragStart"
    @dragenter.prevent="emit('dragenter')"
    @dragover.prevent
    @dragend="onDragEnd"
  >
    <div class="qr__bar">
      <Tooltip label="Drag to reorder" side="top">
        <button
          class="qr__grip"
          type="button"
          aria-label="Drag to reorder"
          @pointerdown="grab = true"
          @pointerup="grab = false"
          @pointercancel="grab = false"
        >
          <Icon name="drag-handle" :size="14" />
        </button>
      </Tooltip>
      <span class="qr__n">{{ index + 1 }}</span>
      <Select :model-value="q.type" :options="typeOptions" @update:model-value="setType" />
      <label class="qr__id">
        <span class="qr__idlabel">id</span>
        <input
          class="qr__idinput"
          :value="q.id"
          :size="Math.max(8, q.id.length + 1)"
          spellcheck="false"
          autocomplete="off"
          placeholder="one_word"
          @input="emit('patch', { id: input($event) })"
        />
      </label>
      <span v-if="answer" class="qr__ans">
        <i class="qr__dot" :class="`qr__dot--${confidenceBin(answer.confidence)}`" />
        {{ answerLabel(answer) }} <span class="qr__ansp">{{ fmtP(answer.confidence) }}</span>
      </span>
      <Menu>
        <MenuTrigger>
          <button class="pk-icon-btn qr__more" type="button" aria-label="Question actions">
            <Icon name="more-horizontal" :size="16" />
          </button>
        </MenuTrigger>
        <MenuContent align="end" label="Question actions">
          <MenuItem :disabled="index === 0" @select="emit('move', -1)">
            <Icon name="arrow-up" :size="14" /> Move up
          </MenuItem>
          <MenuItem :disabled="index === count - 1" @select="emit('move', 1)">
            <Icon name="arrow-down" :size="14" /> Move down
          </MenuItem>
          <MenuItem @select="emit('duplicate')"><Icon name="copy" :size="14" /> Duplicate</MenuItem>
          <MenuSeparator />
          <MenuItem danger @select="emit('remove')"><Icon name="trash" :size="14" /> Delete</MenuItem>
        </MenuContent>
      </Menu>
    </div>

    <textarea
      class="pk-input qr__ta"
      rows="1"
      spellcheck="true"
      :value="q.instructions"
      placeholder="What to decide about the text"
      @input="emit('patch', { instructions: input($event) })"
    />

    <div v-if="q.type === 'noul'" class="qr__pair">
      <input
        class="pk-input pk-input--sm"
        :value="q.yesMeans"
        placeholder="What yes means (optional)"
        @input="emit('patch', { yesMeans: input($event) })"
      />
      <input
        class="pk-input pk-input--sm"
        :value="q.noMeans"
        placeholder="What no means (optional)"
        @input="emit('patch', { noMeans: input($event) })"
      />
    </div>

    <div v-else-if="q.type === 'choice'" class="qr__list">
      <div v-for="(o, i) in q.options" :key="i" class="qr__item">
        <span class="qr__ord">{{ String.fromCharCode(65 + i) }}</span>
        <input
          class="pk-input pk-input--sm qr__name"
          :value="o.name"
          placeholder="Option"
          @input="setOption(i, 'name', $event)"
        />
        <input
          class="pk-input pk-input--sm qr__desc"
          :value="o.description"
          placeholder="What it means (optional)"
          @input="setOption(i, 'description', $event)"
        />
        <button
          class="pk-icon-btn qr__rm"
          type="button"
          aria-label="Remove option"
          :disabled="q.options.length <= 2"
          @click="removeOption(i)"
        >
          <Icon name="x" :size="14" />
        </button>
      </div>
      <button
        class="pk-btn pk-btn--ghost pk-btn--sm qr__add"
        type="button"
        :disabled="q.options.length >= 26"
        @click="addOption"
      >
        <Icon name="plus" :size="13" /> Add option
      </button>
    </div>

    <div v-else class="qr__list">
      <div v-for="(l, i) in q.levels" :key="i" class="qr__item">
        <span class="qr__ord">{{ i + 1 }}</span>
        <input
          class="pk-input pk-input--sm qr__name qr__name--wide"
          :value="l"
          :placeholder="i === 0 ? 'Lowest level' : i === q.levels.length - 1 ? 'Highest level' : 'Level'"
          @input="setLevel(i, $event)"
        />
        <button
          class="pk-icon-btn qr__rm"
          type="button"
          aria-label="Move level up"
          :disabled="i === 0"
          @click="moveLevel(i, -1)"
        >
          <Icon name="arrow-up" :size="13" />
        </button>
        <button
          class="pk-icon-btn qr__rm"
          type="button"
          aria-label="Move level down"
          :disabled="i === q.levels.length - 1"
          @click="moveLevel(i, 1)"
        >
          <Icon name="arrow-down" :size="13" />
        </button>
        <button
          class="pk-icon-btn qr__rm"
          type="button"
          aria-label="Remove level"
          :disabled="q.levels.length <= 2"
          @click="removeLevel(i)"
        >
          <Icon name="x" :size="14" />
        </button>
      </div>
      <button
        class="pk-btn pk-btn--ghost pk-btn--sm qr__add"
        type="button"
        :disabled="q.levels.length >= 26"
        @click="addLevel"
      >
        <Icon name="plus" :size="13" /> Add level
      </button>
    </div>

    <p v-if="error" class="qr__err" role="alert">{{ error }}</p>
  </div>
</template>

<style scoped>
/* a row steps down to the base background inside the surface card (fields
   on surface cards, rows on base - the form recipe) */
.qr {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px 12px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-base);
}
.qr--bad {
  border-color: var(--pk-status-warning);
}
.qr--drag {
  opacity: 0.45;
}
.qr__bar {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.qr__grip {
  display: inline-flex;
  align-items: center;
  padding: 4px 2px;
  border: 0;
  background: none;
  color: var(--pk-text-muted);
  cursor: grab;
  touch-action: none;
}
.qr__grip:active {
  cursor: grabbing;
}
.qr__n {
  min-width: 1.5ch;
  font-size: var(--pk-font-size-xs);
  font-variant-numeric: tabular-nums;
  color: var(--pk-text-muted);
}
.qr__id {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  flex: 0 1 auto;
  min-width: 0;
  max-width: 50%;
  height: 28px;
  padding: 0 8px 0 10px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-full);
  background: var(--pk-bg-inset);
}
/* the answer this row got in the read on show: label + confidence, the
   same swatch scale the Answers card keys its legend on */
.qr__ans {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  max-width: 40%;
  padding: 0 10px;
  height: 28px;
  border-radius: var(--pk-radius-full);
  background: var(--pk-accent-subtle);
  color: var(--pk-text-primary);
  font-size: var(--pk-font-size-xs);
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.qr__ansp {
  font-family: var(--pk-font-mono);
  font-weight: 400;
  color: var(--pk-text-secondary);
}
.qr__dot {
  flex: none;
  width: 9px;
  height: 9px;
  border-radius: 2px;
  border: 1px solid var(--pk-border-subtle);
}
.qr__dot--0 {
  background: var(--pk-conf-1);
}
.qr__dot--1 {
  background: var(--pk-conf-2);
}
.qr__dot--2 {
  background: var(--pk-conf-3);
}
.qr__dot--3 {
  background: var(--pk-conf-4);
}
.qr__id:focus-within {
  border-color: var(--pk-accent);
}
.qr__idlabel {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.qr__idinput {
  flex: 1;
  min-width: 0;
  border: 0;
  background: transparent;
  color: var(--pk-text-primary);
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  outline: none;
}
.qr__more {
  margin-left: auto;
  width: 28px;
  height: 28px;
}
/* one line that grows with the text (field-sizing where the browser has
   it; a manual resize handle everywhere) - a question is a sentence, and
   two fixed rows per question is what pushed Run below the fold */
.qr__ta {
  height: auto;
  min-height: 34px;
  padding: 7px 10px;
  resize: vertical;
  line-height: 1.5;
  font-family: inherit;
  background: var(--pk-bg-inset);
  field-sizing: content;
}
.qr__pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.qr__pair .pk-input,
.qr__item .pk-input {
  background: var(--pk-bg-inset);
}
.qr__list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.qr__item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.qr__ord {
  min-width: 2ch;
  text-align: right;
  font-family: var(--pk-font-mono);
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
}
.qr__name {
  flex: 0 1 180px;
  min-width: 90px;
}
.qr__name--wide {
  flex: 1;
}
.qr__desc {
  flex: 1;
  min-width: 0;
}
.qr__rm {
  width: 26px;
  height: 26px;
  flex: none;
}
.qr__rm:disabled {
  opacity: 0.35;
  cursor: default;
}
.qr__add {
  align-self: flex-start;
  margin-left: 2ch;
}
.qr__err {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-status-warning);
}
@media (max-width: 560px) {
  .qr__pair {
    grid-template-columns: 1fr;
  }
}
</style>
