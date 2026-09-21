<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { syntaxHighlighter } from '@/lib/markdown/runtime'
import { isCancelled, RendererBusyError } from '@/lib/markdown/worker-rpc'
import { copyText } from '@/lib/clipboard'

const props = defineProps<{ node: { code?: string; language?: string; diff?: boolean; raw?: string }; isDark?: boolean; loading?: boolean }>()
// The parser's diff `code` is only the updated side. Preserve the original
// unified patch so removed lines do not silently disappear from display/copy.
const code = computed(() => props.node.diff ? props.node.raw ?? props.node.code ?? '' : props.node.code ?? '')
const language = computed(() => props.node.diff ? 'diff' : props.node.language ?? 'text')
const html = ref('')
const detail = ref('')
const copied = ref('Copy')
const retry = ref(0)
const retryCurrentProps = () => { retry.value++ }
const owner = syntaxHighlighter.open()
let revision = 0
let disposed = false
let copyTimer: ReturnType<typeof setTimeout> | undefined
watch(() => [code.value, language.value, props.isDark, props.loading, retry.value], async () => {
  const current = ++revision
  syntaxHighlighter.cancel(owner); html.value = ''; detail.value = ''
  // Incomplete fences remain selectable plain code. Tokenize once closed,
  // without repainting every token span as each partial line arrives.
  if (props.loading) return
  const source = code.value
  if (source.length > 128 * 1024) { detail.value = 'Large block shown without highlighting'; return }
  try {
    const output = await syntaxHighlighter.submit(owner, { code: source, language: language.value, dark: !!props.isDark }, (source.length + language.value.length) * 2 + 256)
    // Only escaped Shiki output from our bundled worker can reach v-html.
    if (current === revision) html.value = output
  } catch (error) {
    if (current !== revision || isCancelled(error)) return
    if (error instanceof RendererBusyError && syntaxHighlighter.whenAvailable(owner, retryCurrentProps)) return
    detail.value = error instanceof Error ? error.message : 'Highlighting unavailable'
  }
}, { immediate: true })
async function copy() {
  clearTimeout(copyTimer)
  const focused = document.activeElement
  try { await copyText(code.value); if (!disposed) copied.value = 'Copied' }
  catch { if (!disposed) copied.value = 'Copy failed' }
  if (!disposed) {
    if (focused instanceof HTMLElement && focused.isConnected) focused.focus({ preventScroll: true })
    copyTimer = setTimeout(() => { copied.value = 'Copy' }, 2000)
  }
}
onBeforeUnmount(() => { disposed = true; revision++; clearTimeout(copyTimer); syntaxHighlighter.close(owner) })
</script>

<template>
  <div class="pk-code code-block-content">
    <div class="pk-code__header"><span>{{ language }}</span><span v-if="detail" class="pk-code__detail" role="status">{{ detail }}</span><button type="button" aria-label="Copy code" @click="copy">{{ copied }}</button></div>
    <div v-if="html" v-html="html" />
    <pre v-else><code>{{ code }}</code></pre>
  </div>
</template>

<style scoped>
.pk-code { border: 1px solid var(--code-border, #8885); border-radius: var(--pk-radius-lg); margin: var(--ms-flow-codeblock-y, .75em) 0; overflow: hidden; background: var(--code-bg, transparent); }
.pk-code__header { display: flex; gap: 12px; align-items: center; padding: 6px 12px; background: var(--code-header-bg, #8881); font: 12px var(--ms-font-mono, monospace); }
.pk-code__header button { margin-left: auto; color: inherit; cursor: pointer; border: 1px solid transparent; border-radius: var(--pk-radius-md); padding: 3px 8px; background: transparent; font: inherit; }
.pk-code__header button:hover { background: var(--pk-bg-hover); }
.pk-code__header button:focus-visible { outline: 2px solid currentColor; }
.pk-code__detail { font-size: 11px; opacity: .75; }
.pk-code :deep(pre) { margin: 0; padding: 12px; overflow-x: auto; background: transparent !important; font: 12px/1.6 var(--ms-font-mono, ui-monospace, monospace); tab-size: 4; }
</style>
