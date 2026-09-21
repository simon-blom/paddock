<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import MarkdownContent from '@/components/chat/MarkdownContent.vue'
import type { Message } from '@/types/chat'

type DisplayMessage = Pick<Message, 'id' | 'role' | 'content' | 'reasoning' | 'streaming' | 'model' | 'error' | 'stopped' | 'incomplete' | 'usage'> & { nativeStatus?: string; nativeTextParts?: Record<string,string>; nativeReasoningParts?: Record<string,string> }
interface Document { id: string; title: string; messages: DisplayMessage[] }
interface Delta { kind: 'text' | 'reasoning'; outputIndex: number; contentIndex: number; delta: string }
const messages = ref<DisplayMessage[]>([]), dark = ref(false), failure = ref('')
let id: string | undefined, follow = true, observer: ResizeObserver | undefined
const text = new Map<string, string>(), reasoning = new Map<string, string>()
const plain = (m: DisplayMessage) => m.content.filter(p => p.type === 'text').map(p => p.text).join('\n')
const scroll = () => { follow = document.documentElement.scrollHeight - window.scrollY - innerHeight < 72 }
const bottom = () => { if (follow && !window.getSelection()?.toString()) window.scrollTo(0, document.documentElement.scrollHeight) }
const api = {
  async replace(value: Document | null, isDark: boolean) {
    if (value && (!Array.isArray(value.messages) || value.messages.length > 10000)) throw new Error('Invalid transcript document')
    const changed = id !== value?.id
    id = value?.id; text.clear(); reasoning.clear(); failure.value = ''
    const active = value?.messages.at(-1)
    for (const [key, value] of Object.entries(active?.nativeTextParts ?? {})) text.set(key, value)
    for (const [key, value] of Object.entries(active?.nativeReasoningParts ?? {})) reasoning.set(key, value)
    // Preserve keyed message DOM on terminal updates; no document reload or
    // remount of earlier messages, selections, open reasoning or code blocks.
    const previous = new Map(messages.value.map(m => [m.id, m]))
    messages.value = (value?.messages ?? []).map(m => {
      const old = previous.get(m.id)
      return old && JSON.stringify(old) === JSON.stringify(m) ? old : m
    })
    api.theme(isDark)
    if (changed) follow = true
    await nextTick(); bottom()
  },
  async append(events: Delta[], conversationId: string) {
    if (conversationId !== id) return
    if (!Array.isArray(events) || events.length > 4096) throw new Error('Invalid transcript update')
    const message = messages.value.at(-1)
    if (!message || message.role !== 'assistant' || !message.streaming) return
    for (const event of events) {
      if (!['text', 'reasoning'].includes(event.kind) || typeof event.delta !== 'string' || !Number.isSafeInteger(event.outputIndex) || !Number.isSafeInteger(event.contentIndex)) throw new Error('Invalid transcript delta')
      const key = `${String(event.outputIndex).padStart(10, '0')}:${String(event.contentIndex).padStart(10, '0')}`
      const target = event.kind === 'text' ? text : reasoning
      target.set(key, (target.get(key) ?? '') + event.delta)
    }
    const joined = (map: Map<string,string>) => [...map.keys()].sort().map(key => map.get(key)).join('\n')
    message.content = [{ type: 'text', text: joined(text) }]
    message.reasoning = joined(reasoning)
    await nextTick()
  },
  theme(value: boolean) { dark.value = value; document.documentElement.classList.toggle('dark', value) },
}
declare global { interface Window { paddockTranscript: typeof api } }
onMounted(() => {
  window.paddockTranscript = api
  window.addEventListener('scroll', scroll, { passive: true })
  // Progressive Markdown mounting changes height after the data update. Follow
  // those changes only at the bottom, never while the user selects/reads above.
  observer = new ResizeObserver(bottom); observer.observe(document.body)
})
onBeforeUnmount(() => { observer?.disconnect(); window.removeEventListener('scroll', scroll) })
</script>

<template>
  <main aria-label="Conversation transcript">
    <p v-if="failure" class="status" role="status">{{ failure }}</p>
    <article v-for="message in messages" :key="message.id" :data-message-id="message.id" :class="message.role">
      <header>{{ message.role === 'user' ? 'You' : message.model ?? 'Assistant' }}</header>
      <div v-if="message.role !== 'assistant'" class="user-text">{{ plain(message) }}</div>
      <template v-else>
        <details v-if="message.reasoning"><summary>{{ message.streaming ? 'Thinking' : 'Reasoning' }}</summary>
          <MarkdownContent :content="message.reasoning" :streaming="message.streaming" :is-dark="dark" @error="failure = String($event)" />
        </details>
        <MarkdownContent :content="plain(message)" :streaming="message.streaming" :is-dark="dark" @error="failure = String($event)" />
        <p v-if="message.streaming && !plain(message) && !message.reasoning" class="status" role="status">Waiting for response…</p>
      </template>
      <p v-if="message.content.some(p => p.type !== 'text')" class="status">This message includes attachments; native attachment viewing is not connected yet.</p>
      <p v-if="message.error" class="status" role="alert">{{ message.error }}</p>
      <p v-else-if="message.stopped" class="status">Stopped</p>
      <p v-else-if="message.nativeStatus === 'incomplete' || message.incomplete" class="status">Response incomplete</p>
      <footer v-if="message.usage?.completionTokens != null">{{ message.usage.completionTokens }} output tokens</footer>
    </article>
  </main>
</template>
