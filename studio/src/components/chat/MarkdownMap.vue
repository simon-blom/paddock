<script setup lang="ts">
import { computed } from 'vue'
import { parseMapBody } from '@/lib/mapblock'
import type { FileLocation } from '@/lib/api'
import PhotoLocation from './PhotoLocation.vue'
import MarkdownCode from './MarkdownCode.vue'

const props = defineProps<{ node: { code?: string; language?: string }; loading?: boolean; isDark?: boolean }>()
const location = computed<FileLocation | null>(() => {
  if (props.loading) return null
  const map = parseMapBody(props.node.code ?? '')
  if (!map) return null
  return { latitude: map.lat, longitude: map.lon, place: map.label ? {
    city: map.label, region: '', country: '', distance_km: 0, bearing: 'N', description: map.label,
  } : undefined }
})
</script>
<template>
  <div v-if="location" class="pk-md__map"><PhotoLocation :location="location" compact /></div>
  <MarkdownCode v-else-if="!loading" :node="node" :is-dark="isDark" />
  <span v-else aria-label="Map coordinates arriving" />
</template>
<style scoped>
.pk-md__map { max-width: 420px; margin: 10px 0; }
</style>
