<script setup lang="ts">
// The only mounted web UI in the native app: Lector, Scriptor and Traverse.
// Conversation effects are shared data logic, not ChatView/ChatThread.
import { computed, defineAsyncComponent, inject, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { NATIVE_CONTENT } from '@/lib/native-content'
import { useChatStore } from '@/stores/chat'
import { useModelsStore } from '@/stores/models'
import { useGraphsStore } from '@/stores/graphs'
import { useArtifactsStore } from '@/stores/artifacts'
import type { ImportResult } from '@/lib/graph/session'
import { useConversationEffects } from '@/composables/useConversationEffects'
import { isDocxPart, isPdfPart } from '@/lib/docrun'
import { TooltipProvider } from 'reka-ui'
const DocumentPane = defineAsyncComponent(() => import('@/components/chat/DocumentPane.vue'))
const GraphPane = defineAsyncComponent(() => import('@/components/chat/graph/GraphPane.vue'))
const GraphArtifact = defineAsyncComponent(() => import('@/components/chat/graph/GraphArtifact.vue'))
const native = inject(NATIVE_CONTENT)!
const chat = useChatStore(), models = useModelsStore(), graphs = useGraphsStore()
const artifacts = useArtifactsStore()
function graphResult(result: { errors: ImportResult['errors']; executed: number }) {
  const meta = artifacts.list.find(a => a.id === native.graphArtifact?.value?.id)
  if (meta) artifacts.reportGraphImport(meta.id, meta.versions, result)
}
const route = useRoute(), router = useRouter()
const pane = ref<InstanceType<typeof DocumentPane> | null>(null)
const document = computed(() => {
  const c = native.document?.value
  return c?.messages.some(m => m.content.some(p => p.type === 'file' && (isPdfPart(p) || isDocxPart(p)))) ? c : null
})
useConversationEffects()
watch(pane, value => native.documentActions?.(value), { flush: 'post' })
onBeforeUnmount(() => native.documentActions?.(null))
async function syncRoute() {
  if (route.name === 'home' || route.name === 'chat-new') {
    chat.startDraft(models.currentId || 'default'); return
  }
  const requested = typeof route.params.id === 'string' ? route.params.id : ''
  const id = chat.conversations.some(c => c.id === requested) ? requested : chat.lastOpenId()
  if (!id) { await router.replace({ name: 'home' }); return }
  chat.select(id)
  if (id !== requested) await router.replace({ name: 'chat', params: { id } })
}
onMounted(async () => {
  await chat.hydrate()
  void models.refresh()
  await syncRoute()
  native.mounted()
})
watch(() => [route.name, route.params.id], () => void syncRoute())
</script>
<template>
  <div class="native-embedded-viewers" data-native-surface="embedded-viewers">
    <TooltipProvider>
      <DocumentPane v-if="document" ref="pane" :key="document.id" :conversation="document" embedded class="native-document-surface" />
      <GraphArtifact v-else-if="native.graphArtifact?.value" :key="native.graphArtifact.value.id" :content="native.graphArtifact.value.body" :title="native.graphArtifact.value.title" @import-result="graphResult" />
      <GraphPane v-else-if="native.graphVisible?.value && graphs.active && !graphs.folded" />
    </TooltipProvider>
  </div>
</template>
<style scoped>
.native-embedded-viewers { display: flex; width: 100%; height: 100%; min-width: 0; overflow: hidden; }
.native-document-surface { width: 100%; min-width: 0; max-width: none; flex: 1; border: 0; }
</style>
