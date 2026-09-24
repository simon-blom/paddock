<script setup lang="ts">
// The picture popover - the sampler's seat when the composer is in image
// mode. Size, quality, steps, seed, count, format, background and previews,
// per conversation and persisted, with every turn recording what rode (Run
// details). The trigger is the parent's slot content, exactly as SamplerMenu
// takes it (see the popper note there).
//
// The controls are drawn from what the ENDPOINT advertises (its grid, its
// ceiling, its formats, whether it streams previews), never from a house
// list - a second image family with a different grid gets its own controls
// without a change here. Automatic seeds preserve text-only variations;
// reference edits and retries draw afresh. Explicit pinning stays available.
import { computed, watch } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useModelsStore } from '@/stores/models'
import { DEFAULT_IMAGE_PARAMS, imageParamsOf, type ImageParams } from '@/types/chat'
import Menu from '@/components/ui/Menu.vue'
import MenuContent from '@/components/ui/MenuContent.vue'
import NumberField from '@/components/ui/NumberField.vue'
import Select, { type SelectOption } from '@/components/ui/Select.vue'
import TextInput from '@/components/ui/TextInput.vue'

const props = defineProps<{
  /** the lanes this send goes to - the first one's advertised limits shape
   *  the controls (a compare set shares one prompt and one seed) */
  lanes: string[]
}>()

const chat = useChatStore()
const models = useModelsStore()

watch(
  () => props.lanes,
  (ids) => {
    for (const id of ids) void models.capsFor(id)
  },
  { immediate: true },
)
const caps = computed(() => models.caps[props.lanes[0] ?? '']?.imageGeneration)

const params = computed<ImageParams>(() =>
  chat.active ? imageParamsOf(chat.active) : { ...DEFAULT_IMAGE_PARAMS },
)
function set<K extends keyof ImageParams>(key: K, value: ImageParams[K]): void {
  const c = chat.active
  if (!c) return
  c.imageParams = { ...imageParamsOf(c), [key]: value }
  chat.persist(c)
}

// ── size ─────────────────────────────────────────────────────────────────
// The three shapes the OpenAI API names, the model's own default, and a
// custom pair on the endpoint's grid. Anything stored that is not a preset
// reads as custom, so a size picked by hand survives a reopen.
const PRESETS = ['auto', '1024x1024', '1536x1024', '1024x1536'] as const
const sizeOptions = computed<SelectOption[]>(() => [
  { value: 'auto', label: 'Model default', hint: caps.value?.defaultSize },
  { value: '1024x1024', label: 'Square', hint: '1024 x 1024' },
  { value: '1536x1024', label: 'Landscape', hint: '1536 x 1024' },
  { value: '1024x1536', label: 'Portrait', hint: '1024 x 1536' },
  { value: 'custom', label: 'Custom', hint: 'width x height' },
])
const sizePick = computed<string>({
  get: () => ((PRESETS as readonly string[]).includes(params.value.size) ? params.value.size : 'custom'),
  set: (v) => {
    if (v === 'custom') {
      // start the pair from what is there, or the model's default
      const cur = (PRESETS as readonly string[]).includes(params.value.size)
        ? (caps.value?.defaultSize ?? '1024x1024')
        : params.value.size
      set('size', cur === 'auto' ? '1024x1024' : cur)
    } else {
      set('size', v)
    }
  },
})
const custom = computed(() => sizePick.value === 'custom')
const grid = computed(() => caps.value?.sizeMultiple ?? 32)
const maxSide = computed(() => caps.value?.maxSide ?? 2048)
function side(i: 0 | 1) {
  return computed<number>({
    get: () => {
      const m = /^(\d+)x(\d+)$/.exec(params.value.size)
      return m ? Number(m[i + 1]) : 1024
    },
    set: (v) => {
      // snap onto the grid the endpoint refuses to leave
      const g = grid.value
      const snapped = Math.min(maxSide.value, Math.max(g, Math.round(v / g) * g))
      const w = i === 0 ? snapped : side(0).value
      const h = i === 1 ? snapped : side(1).value
      set('size', `${w}x${h}`)
    },
  })
}
const width = side(0)
const height = side(1)

// ── quality / steps ──────────────────────────────────────────────────────
const qualityOptions = computed<SelectOption[]>(() => [
  { value: 'auto', label: 'Model default', hint: caps.value ? `${caps.value.defaultSteps} steps` : undefined },
  { value: 'low', label: 'Low', hint: 'fewer steps, faster' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High', hint: caps.value ? `${caps.value.defaultSteps} steps` : undefined },
])
const quality = computed<ImageParams['quality']>({
  get: () => params.value.quality,
  set: (v) => set('quality', v),
})
const stepsText = computed<string>({
  get: () => (params.value.steps == null ? '' : String(params.value.steps)),
  set: (v) => {
    const n = Number.parseInt(v, 10)
    const max = caps.value?.maxSteps ?? 100
    set('steps', Number.isFinite(n) && n > 0 ? Math.min(n, max) : null)
  },
})

// ── seed ─────────────────────────────────────────────────────────────────
// Keep the persisted 'thread' value for compatibility. For edits, the
// reference preserves composition and a fresh seed avoids noise reuse.
const seedOptions: SelectOption[] = [
  { value: 'thread', label: 'Automatic' },
  { value: 'random', label: 'New every picture' },
  { value: 'pinned', label: 'Pinned', hint: 'this number, every time' },
]
const seedMode = computed<string>({
  get: () => (typeof params.value.seed === 'number' ? 'pinned' : params.value.seed),
  set: (v) => {
    if (v === 'pinned') {
      // pin what the conversation is on, or draw one to pin
      set('seed', lastSeed() ?? Math.floor(Math.random() * 2 ** 31))
    } else {
      set('seed', v as 'thread' | 'random')
    }
  },
})
function lastSeed(): number | undefined {
  const msgs = chat.active?.messages ?? []
  for (let i = msgs.length - 1; i >= 0; i--) {
    const s = msgs[i].imageGen?.seed
    if (s !== undefined) return s
  }
  return undefined
}
const pinned = computed(() => seedMode.value === 'pinned')
const seedText = computed<string>({
  get: () => (params.value.seed === 'random' ? '' : String(params.value.seed)),
  set: (v) => {
    const n = Number.parseInt(v, 10)
    if (Number.isFinite(n) && n >= 0) set('seed', n)
  },
})

// ── count / format / background / previews ───────────────────────────────
const countOptions = computed<SelectOption[]>(() => {
  const max = Math.min(4, caps.value?.maxN ?? 1)
  return Array.from({ length: max }, (_, i) => ({
    value: i + 1,
    label: i === 0 ? '1 picture' : `${i + 1} pictures`,
    hint: i === 0 ? undefined : 'each its own draw on the seed',
  }))
})
const count = computed<number>({
  get: () => params.value.n,
  set: (v) => set('n', Number(v)),
})
const formatOptions = computed<SelectOption[]>(() =>
  (caps.value?.outputFormats ?? ['png', 'webp', 'jpeg']).map((f) => ({
    value: f,
    label: f.toUpperCase(),
    hint: f === 'jpeg' ? 'no transparency' : f === 'png' ? 'lossless' : undefined,
  })),
)
const format = computed<ImageParams['format']>({
  get: () => params.value.format,
  set: (v) => set('format', v),
})
const backgroundOptions: SelectOption[] = [
  { value: 'auto', label: 'As the model decides' },
  { value: 'opaque', label: 'Opaque' },
  { value: 'transparent', label: 'Transparent', hint: 'PNG or WebP' },
]
const background = computed<ImageParams['background']>({
  get: () => params.value.background,
  set: (v) => set('background', v),
})
const transparentOnJpeg = computed(
  () => params.value.background === 'transparent' && params.value.format === 'jpeg',
)
const previewOptions = computed<SelectOption[]>(() => {
  const max = caps.value?.stream ? (caps.value.maxPartialImages ?? 0) : 0
  return Array.from({ length: max + 1 }, (_, i) => ({
    value: i,
    label: i === 0 ? 'Off' : `${i} while rendering`,
    hint: i === 0 ? 'the finished picture only' : undefined,
  }))
})
const previews = computed<number>({
  get: () => Math.min(params.value.previews, previewOptions.value.length - 1),
  set: (v) => set('previews', Number(v)),
})
const previewsOff = computed(() => params.value.n > 1)

const changed = computed(() => {
  const p = params.value
  return (Object.keys(DEFAULT_IMAGE_PARAMS) as (keyof ImageParams)[]).some(
    (k) => p[k] !== DEFAULT_IMAGE_PARAMS[k],
  )
})
function reset(): void {
  const c = chat.active
  if (!c) return
  c.imageParams = undefined
  chat.persist(c)
}
</script>

<template>
  <Menu>
    <slot />
    <MenuContent side="top" align="start" min-width="340px">
      <div class="io__body">
        <div class="io__row">
          <span class="io__label">Size</span>
          <span class="io__pick"><Select v-model="sizePick" :options="sizeOptions" block /></span>
        </div>
        <div v-if="custom" class="io__row io__row--pair">
          <NumberField v-model="width" :min="grid" :max="maxSide" :step="grid" />
          <span class="io__x">x</span>
          <NumberField v-model="height" :min="grid" :max="maxSide" :step="grid" />
        </div>
        <p v-if="custom" class="io__note">Multiples of {{ grid }}, up to {{ maxSide }} a side.</p>
        <div class="io__row">
          <span class="io__label">Quality</span>
          <span class="io__pick"><Select v-model="quality" :options="qualityOptions" block /></span>
        </div>
        <div class="io__row">
          <span class="io__label">Steps</span>
          <span class="io__num">
            <TextInput v-model="stepsText" placeholder="quality's" />
          </span>
        </div>
        <div class="io__row">
          <span class="io__label">Seed</span>
          <span class="io__pick"><Select v-model="seedMode" :options="seedOptions" block /></span>
        </div>
        <div v-if="pinned" class="io__row">
          <span class="io__label">Number</span>
          <span class="io__num">
            <TextInput v-model="seedText" placeholder="0" />
          </span>
        </div>
        <div class="io__row">
          <span class="io__label">Count</span>
          <span class="io__pick"><Select v-model="count" :options="countOptions" block /></span>
        </div>
        <div class="io__row">
          <span class="io__label">Format</span>
          <span class="io__pick"><Select v-model="format" :options="formatOptions" block /></span>
        </div>
        <div class="io__row">
          <span class="io__label">Background</span>
          <span class="io__pick"><Select v-model="background" :options="backgroundOptions" block /></span>
        </div>
        <p v-if="transparentOnJpeg" class="io__note io__note--warn">
          A transparent background needs PNG or WebP - JPEG has no alpha channel.
        </p>
        <div v-if="previewOptions.length > 1" class="io__row">
          <span class="io__label">Previews</span>
          <span class="io__pick">
            <Select v-model="previews" :options="previewOptions" block :disabled="previewsOff" />
          </span>
        </div>
        <p v-if="previewsOff && previewOptions.length > 1" class="io__note">
          Previews show one picture at a time - off while several are asked for.
        </p>
        <button v-if="changed" class="pk-btn pk-btn--sm io__reset" type="button" @click="reset">
          Defaults
        </button>
      </div>
    </MenuContent>
  </Menu>
</template>

<style scoped>
.io__body {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  width: 328px;
}
.io__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.io__row--pair {
  justify-content: flex-start;
}
.io__label {
  font-size: var(--pk-font-size-sm);
  color: var(--pk-text-secondary);
}
/* The control column. A class on <Select> itself never lands: its root is
   Reka's SelectRoot, a fragment, so the trigger kept the primitive's 220px
   floor and every row ran past the panel. The width lives on this wrapper
   and the trigger fills it (`block`). */
.io__pick {
  flex: none;
  width: 176px;
  min-width: 0;
}
.io__pick :deep(.pk-select) {
  min-width: 0;
}
.io__x {
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-sm);
}
.io__num {
  width: 110px;
  min-width: 0;
}
.io__num :deep(input) {
  width: 100%;
  min-width: 0;
  text-align: right;
  font-family: var(--pk-font-mono);
}
.io__note {
  margin: 2px 0 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-muted);
  max-width: 100%;
}
.io__note--warn {
  color: var(--pk-text-danger);
}
.io__reset {
  align-self: flex-start;
}
</style>
