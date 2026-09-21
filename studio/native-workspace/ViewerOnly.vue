<script setup lang="ts">
// No chat store, router, stream controller or media engine. Swift supplies an
// isolated document projection; this page owns only the three allowed viewers.
import { defineAsyncComponent, onBeforeUnmount, ref, shallowRef } from 'vue'
import type { Conversation, GraphPart } from '@/types/chat'
import { useGraphsStore } from '@/stores/graphs'
import { useSettingsStore } from '@/stores/settings'
import { TooltipProvider } from 'reka-ui'
const DocumentPane = defineAsyncComponent(() => import('@/components/chat/DocumentPane.vue'))
const GraphPane = defineAsyncComponent(() => import('@/components/chat/graph/GraphPane.vue'))
const GraphArtifact = defineAsyncComponent(() => import('@/components/chat/graph/GraphArtifact.vue'))
const document = shallowRef<Conversation | null>(null)
const graph = shallowRef<{ id: string; title: string; body: string } | null>(null)
const pane = ref<InstanceType<typeof DocumentPane> | null>(null)
const graphs = useGraphsStore(), settings = useSettingsStore()
const visibleGraph = ref(false)
let epoch = 0
async function update(value: { document?: Conversation; graph?: typeof graph.value; graphSource?: GraphPart; conversationId: string; visibleGraph?: boolean }, dark: boolean) {
  const ticket = ++epoch
  settings.theme = dark ? 'dark' : 'light'
  window.document.documentElement.dataset.theme = settings.theme
  document.value = value.document ?? null; graph.value = value.graph ?? null
  visibleGraph.value = value.visibleGraph === true
  if (value.graphSource) {
    if (!/^[a-zA-Z0-9_-]{1,128}$/.test(value.graphSource.attachmentId) || !/^[a-zA-Z0-9_-]{1,128}$/.test(value.conversationId)) throw new Error('Invalid graph scope')
    await graphs.ensure(value.conversationId, value.graphSource.attachmentId, value.graphSource.name)
    if (ticket !== epoch) return ''
    if (graphs.status !== 'ready') throw new Error(graphs.error || 'The graph could not be opened')
    graphs.folded = false
    return graphs.groundingFor(value.conversationId)
  }
  graphs.release()
  return ''
}
window.paddockViewer = {
  update,
  action(name: string) {
    if (!pane.value) throw new Error('Open a document first')
    if (name === 'info') pane.value.info()
    else if (name === 'download') pane.value.download()
    else throw new Error('Unknown viewer action')
  },
  close() { ++epoch; document.value = null; graph.value = null; graphs.release() },
}
onBeforeUnmount(() => window.paddockViewer.close())
</script>
<template>
  <div class="native-embedded-viewers" data-native-surface="viewer-only">
    <TooltipProvider>
      <DocumentPane v-if="document" ref="pane" :key="document.id" :conversation="document" embedded class="native-document-surface" />
      <GraphArtifact v-else-if="graph" :key="graph.id" :content="graph.body" :title="graph.title" />
      <GraphPane v-else-if="visibleGraph && graphs.active" />
    </TooltipProvider>
  </div>
</template>
<style scoped>
.native-embedded-viewers { display: flex; width: 100%; height: 100%; min-width: 0; overflow: hidden; }
.native-document-surface { width: 100%; min-width: 0; max-width: none; flex: 1; border: 0; }
</style>
