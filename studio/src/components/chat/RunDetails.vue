<script setup lang="ts">
import { computed } from 'vue'
import type { Message } from '@/types/chat'
import { runDetailSections } from '@/lib/message-presentation'
import Collapsible from '@/components/ui/Collapsible.vue'

const props = defineProps<{ message: Message }>()
const sections = computed(() => runDetailSections(props.message))
</script>

<template>
  <div v-if="sections.length" class="run">
    <div v-for="section in sections" :key="section.id" class="run__sec">
      <span class="run__title">{{ section.title }}</span>
      <dl class="run__grid">
        <template v-for="row in section.rows" :key="row.label">
          <dt>{{ row.label }}</dt>
          <dd>{{ row.value }}</dd>
        </template>
      </dl>
      <Collapsible v-if="section.id === 'provenance' && message.run?.systemPrompt" class="run__prompt" summary="Prompt text">
        <pre>{{ message.run.systemPrompt }}</pre>
      </Collapsible>
    </div>
  </div>
</template>

<style scoped>
.run {
  margin-top: 8px;
  padding: 12px 14px;
  border: 1px solid var(--pk-border-default);
  border-radius: var(--pk-radius-lg);
  background: var(--pk-bg-base);
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.run__sec {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.run__title {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--pk-text-muted);
}
.run__grid {
  display: grid;
  grid-template-columns: 96px 1fr;
  gap: 3px 12px;
  margin: 0;
}
.run__grid dt {
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-secondary);
}
.run__grid dd {
  margin: 0;
  font-size: var(--pk-font-size-xs);
  color: var(--pk-text-primary);
  font-family: var(--pk-font-mono);
  word-break: break-word;
}
.run__prompt {
  margin-top: 2px;
}
.run__prompt pre {
  margin: 6px 0 0;
  padding: 8px 10px;
  border-radius: var(--pk-radius-md);
  background: var(--pk-bg-inset);
  font-size: var(--pk-font-size-xs);
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--pk-text-secondary);
  max-height: 220px;
  overflow-y: auto;
}
</style>
