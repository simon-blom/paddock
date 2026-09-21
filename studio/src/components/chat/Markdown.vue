<script setup lang="ts">
// Product settings/maps live here. The store-free rendering engine is shared
// unchanged with the isolated WebKit lab.
import { computed } from 'vue'
import MarkdownContent from './MarkdownContent.vue'
import { useSettingsStore } from '@/stores/settings'
import { initMarkdownMaps } from '@/lib/markstream-maps'

defineProps<{ content: string; streaming?: boolean }>()
initMarkdownMaps()
const settings = useSettingsStore()
const isDark = computed(() => settings.theme === 'dark')
</script>

<template>
  <MarkdownContent :content="content" :streaming="streaming" :is-dark="isDark" custom-id="paddock-rich-maps" />
</template>
