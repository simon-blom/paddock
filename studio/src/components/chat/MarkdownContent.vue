<script setup lang="ts">
// Store-free production renderer shared with the native WebKit lab. Parsing
// runs off-thread; immutable groups keep settled history out of Vue updates.
import { markRaw, nextTick, onBeforeUnmount, ref, shallowRef, watch } from 'vue'
import { MarkdownRender } from 'markstream-vue'
import type { ParsedNode } from 'stream-markdown-parser'
import { initMarkstream } from '@/lib/markstream'
import { markdownParser } from '@/lib/markdown/runtime'
import { applyPatch, groupNodes, MAX_MARKDOWN_CHARS } from '@/lib/markdown/parse'
import { isCancelled, RendererBusyError } from '@/lib/markdown/worker-rpc'
import { markdownMounts } from '@/lib/markdown/mount-scheduler'

const props = defineProps<{ content: string; streaming?: boolean; isDark?: boolean; customId?: string }>()
const emit = defineEmits<{ error: [error: unknown] }>()
initMarkstream()
// A mounted empty view is not a parser consumer. In particular, an empty
// sibling must not keep a populated view's worker/VM alive after it closes.
// Populated consumers still share the same worker and its normal idle policy.
const owner = shallowRef<string | null>(null)
const groups = shallowRef<{ nodes: ParsedNode[]; final: boolean }[]>([])
let target: ParsedNode[][] = []
let parsedRevision = 0
const fallback = ref('')
const ready = ref(false)
const appliedFinal = ref(false)
const retry = ref(0)
const retryCurrentProps = () => { retry.value++ }
let nodes: ParsedNode[] = []
let requested = 0
const applied = ref(0)
const mermaidProps = { onRenderError: (error: unknown) => emit('error', error) }

function releaseParser() {
  if (owner.value === null) return
  markdownMounts.cancel(owner.value)
  markdownParser.close(owner.value)
  owner.value = null
}

watch(() => [props.content, props.streaming, retry.value] as const, async ([content, streaming]) => {
  const revision = ++requested
  const final = !streaming
  if (owner.value !== null) markdownParser.cancel(owner.value)
  fallback.value = ''
  if (!content) {
    releaseParser(); nodes = []; target = []; groups.value = []
    parsedRevision = 0; applied.value = revision; appliedFinal.value = final; ready.value = true; return
  }
  if (content.length > MAX_MARKDOWN_CHARS) {
    releaseParser(); groups.value = []; target = []; nodes = []; parsedRevision = 0
    fallback.value = 'Large message shown as complete plain text'; return
  }
  // Capture this lease across awaits: clearing/reopening can create a new
  // owner before the old worker reply or mount quantum has finished.
  const parserOwner = owner.value ?? (owner.value = markdownParser.open())
  try {
    const patch = await markdownParser.submit(parserOwner, { content, final, revision, base: parsedRevision }, content.length * 2 + 256)
    if (revision !== requested) return
    nodes = applyPatch(nodes, parsedRevision, patch); parsedRevision = revision
    // Fewer renderer instances reduce per-group parser/component overhead.
    // The shared scheduler still charges the actual flush for each group.
    target = groupNodes(nodes, target, 64)
    let cursor = 0
    const fail = (error: unknown) => {
      if (revision !== requested) return
      fallback.value = error instanceof Error ? error.message : 'Rich rendering unavailable; showing complete plain text'
      emit('error', error)
    }
    const eager = groups.value.length > 0 && target.length <= groups.value.length + 1
      && target.filter((group, i) => group !== groups.value[i]?.nodes || final !== groups.value[i]?.final).length <= 1
    // A one-group warm edit commits atomically. Keep the previous revision
    // readable until then instead of issuing an extra whole-parent Vue flush
    // just to toggle busy on and off for every incoming token.
    if (!eager) ready.value = false
    markdownMounts.schedule(parserOwner, { error: fail, step: async () => {
      // The parser may finish another revision while this quantum flushes.
      // Only that new plan may continue; already mounted groups stay in place.
      if (revision !== parsedRevision) return false
      while (cursor < target.length && groups.value[cursor]?.nodes === target[cursor] && groups.value[cursor]?.final === final) cursor++
      if (cursor < target.length) {
        const next = groups.value.slice(0, target.length)
        next[cursor] = { nodes: target[cursor], final }; cursor++
        groups.value = markRaw(next)
      } else if (groups.value.length !== target.length) groups.value = groups.value.slice(0, target.length)
      const complete = cursor === target.length
      if (complete) { applied.value = revision; appliedFinal.value = final; ready.value = true }
      await nextTick()
      return !complete && revision === parsedRevision
    } }, eager)
  } catch (error) {
    if (revision !== requested || isCancelled(error)) return
    if (error instanceof RendererBusyError && markdownParser.whenAvailable(parserOwner, retryCurrentProps)) return
    markdownMounts.cancel(parserOwner)
    fallback.value = error instanceof Error ? error.message : 'Rich rendering unavailable; showing complete plain text'
    emit('error', error)
  }
}, { immediate: true })
onBeforeUnmount(() => { requested++; parsedRevision = -1; releaseParser() })
</script>

<template>
  <div class="pk-md markstream-vue markdown-renderer" :class="{ dark: isDark }" :aria-busy="!ready && !fallback" :data-markdown-revision="applied" :data-markdown-final="appliedFinal" :data-markdown-state="fallback ? 'plain' : ready ? 'rich' : 'pending'">
    <template v-if="fallback">
      <small role="status">{{ fallback }}</small>
      <pre class="pk-md__plain">{{ content }}</pre>
    </template>
    <!-- Do not lay out the entire raw document only to throw that DOM away
         milliseconds later. Readiness stays pending until ALL rich groups
         commit; the lab separately measures first and complete rich content. -->
    <span v-else-if="!ready && !groups.length" role="status" class="pk-md__pending">Rendering message…</span>
    <template v-else>
      <MarkdownRender v-for="(group, i) in groups" :key="i" :custom-id="customId ?? 'paddock-rich'"
        :nodes="group.nodes" :index-key="`${owner}-${i}`" :is-dark="isDark" :final="group.final"
        :mermaid-props="mermaidProps" :render-code-blocks-as-pre="false" html-policy="escape"
        :typewriter="false" :smooth-streaming="false" :batch-rendering="false"
        :render-as-fragment="true" :defer-nodes-until-visible="false" :node-virtual="false" mode="chat" />
    </template>
  </div>
</template>

<style scoped>
.pk-md__plain { white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; margin: 0; }
.pk-md__pending { color: var(--pk-text-muted); font-size: .85em; }
</style>
