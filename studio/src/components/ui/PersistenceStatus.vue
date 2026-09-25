<script setup lang="ts">
import { uiPreferences } from '@/lib/ui-preferences'
const props = defineProps<{ migrationError?: string; retryMigration?: () => Promise<void> }>()
const error = uiPreferences.error
const retry = () => { void uiPreferences.flush().catch(() => {}); void props.retryMigration?.() }
</script>
<template>
  <div v-if="error || migrationError" class="persistence-error" role="alert">
    <span>{{ error ? `${error}. Changes are not saved.` : migrationError }}</span>
    <button class="pk-btn" @click="retry">Retry</button>
  </div>
</template>
<style scoped>
.persistence-error { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); z-index: 9999; display: flex; align-items: center; gap: 12px; padding: 12px 16px; max-width: calc(100% - 32px); background: var(--pk-bg-elevated); color: var(--pk-text-primary); border: 1px solid var(--pk-border-default); border-radius: var(--pk-radius-lg); }
</style>
