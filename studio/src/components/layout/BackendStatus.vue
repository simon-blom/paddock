<script setup lang="ts">
import { useReadinessStore } from '@/stores/readiness'
import Icon from '@/components/Icon.vue'
import Tooltip from '@/components/ui/Tooltip.vue'

const readiness = useReadinessStore()
</script>

<template>
  <Tooltip v-if="readiness.headerStatus" :label="readiness.headerStatus.detail" side="bottom">
    <span class="backend-status" tabindex="0" aria-label="Metal backend information">
      <Icon name="graphics-card" :size="13" aria-hidden="true" />
      {{ readiness.headerStatus.label }}
    </span>
  </Tooltip>
</template>

<style scoped>
.backend-status {
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
  gap: 4px;
  padding: 3px 2px;
  border-radius: var(--pk-radius-sm);
  color: var(--pk-text-muted);
  font-size: var(--pk-font-size-xs);
  line-height: 1.2;
  white-space: nowrap;
  cursor: help;
}
.backend-status:hover,
.backend-status:focus-visible {
  color: var(--pk-text-secondary);
}
.backend-status:focus-visible {
  outline: 1px solid var(--pk-border-strong);
  outline-offset: 2px;
}
</style>
