<script setup lang="ts">
import { nextTick, ref } from 'vue'
import MarkdownContent from '@/components/chat/MarkdownContent.vue'
import { useStickyScroll } from '@/composables/useStickyScroll'
import { markdownMounts } from '@/lib/markdown/mount-scheduler'
import { assert, check, delay, until } from './report'

const scroll = ref<HTMLElement | null>(null)
const body = ref<HTMLElement | null>(null)
const content = ref('')
const streaming = ref(true)
const visible = ref(true)
const sticky = useStickyScroll(scroll, body)
const paragraphs = (count: number) => Array.from({ length: count }, (_, i) => `Paragraph ${i}: **stable selection ${i}** and inline \`code\`.\n\n`).join('')
const ready = (count: number) => body.value?.querySelector('[data-markdown-state="rich"]') && body.value.querySelectorAll('strong').length === count

async function run() {
  await check('Frame mounting preserves selection and user scroll', async () => {
    content.value = paragraphs(64); await until(() => ready(64), 'Initial history mount incomplete')
    scroll.value!.scrollTop = 200; await delay(80)
    assert(!sticky.stuck.value, 'User scroll did not release sticky mode')
    const selected = body.value!.querySelectorAll('strong')[4]
    const text = selected.firstChild!
    const range = document.createRange(); range.selectNodeContents(selected)
    const selection = document.getSelection()!; selection.removeAllRanges(); selection.addRange(range)
    const before = scroll.value!.scrollTop
    content.value = paragraphs(320)
    await until(() => ready(320), 'Appended history missing')
    await delay(80)
    assert(selected.isConnected && selected.firstChild === text, 'Settled selected DOM was replaced')
    assert(selection.toString() === 'stable selection 4', 'Text selection was lost during mounting')
    assert(Math.abs(scroll.value!.scrollTop - before) <= 1, 'Mounting moved the unpinned scroll position')
    streaming.value = false; await nextTick()
    await until(() => body.value?.querySelector('[data-markdown-final="true"]'), 'Final mount did not complete')
    assert(selection.toString() === 'stable selection 4' && text.isConnected, 'Finalization lost selection')
    selection.removeAllRanges()
    return '320 rich paragraphs fully present; append/finalization retain the selected DOM node, range and unpinned scrollTop'
  })
  await check('Frame mounting follows sticky scroll and cancels disposal', async () => {
    sticky.toBottom(); content.value = paragraphs(480); streaming.value = true
    await until(() => ready(480), 'Pinned history missing'); await delay(100)
    const el = scroll.value!
    assert(el.scrollHeight - el.scrollTop - el.clientHeight <= 2, 'Pinned reader did not follow incremental mounting')
    // Queue a pin, then scroll away before it executes: the old callback must
    // not pull the user back to the bottom on the next animation frame.
    sticky.schedulePin(); el.scrollTop = 100; el.dispatchEvent(new Event('scroll'))
    await delay(80); assert(Math.abs(el.scrollTop - 100) <= 1, 'Queued sticky pin overrode a later user scroll')
    content.value = paragraphs(1500); await nextTick()
    visible.value = false; await nextTick(); await delay(80)
    assert(markdownMounts.stats.owners === 0, 'Disposed renderer left a mount job behind')
    visible.value = true; content.value = '**Replacement**'; streaming.value = false
    await until(() => ready(1), 'Replacement mount missing')
    assert(body.value!.textContent === 'Replacement', 'An obsolete revision rendered after disposal')
    return 'Sticky bottom follows 480 mounted paragraphs; a later user scroll cancels the queued pin; dispose/reopen discards obsolete jobs'
  })
}
defineExpose({ run })
</script>
<template>
  <div ref="scroll" class="md-scroll-case"><div ref="body"><MarkdownContent v-if="visible" :content="content" :streaming="streaming" /></div></div>
</template>
<style scoped>
.md-scroll-case { height: 240px; overflow: auto; border: 1px solid var(--line); }
</style>
