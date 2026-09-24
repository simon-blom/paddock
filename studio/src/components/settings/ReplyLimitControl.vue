<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import Select from '@/components/ui/Select.vue'
import TextInput from '@/components/ui/TextInput.vue'
import { parseReplyLimit, REPLY_LIMIT_ERROR } from '@/lib/reply-limit'

const model = defineModel<number | null>({ required: true })
const mode = ref<string | number>(model.value == null ? 'automatic' : 'custom')
const text = ref(model.value?.toString() ?? '')
const value = computed(() => mode.value === 'automatic' ? null : parseReplyLimit(text.value))
const invalid = computed(() => mode.value === 'custom' && value.value == null)
const dirty = computed(() => invalid.value || value.value !== model.value)
watch(model, v => {
  mode.value = v == null ? 'automatic' : 'custom'
  if (v != null) text.value = String(v)
})
function apply(): void {
  if (!invalid.value) model.value = value.value
}
</script>

<template>
  <div class="reply-limit">
    <Select v-model="mode" :options="[
      { value: 'automatic', label: 'Automatic' }, { value: 'custom', label: 'Custom' },
    ]" block aria-label="Reply limit" />
    <div v-if="mode === 'custom'" class="reply-limit__entry">
      <TextInput v-model="text" block placeholder="Token limit" inputmode="numeric"
        aria-label="Maximum reply tokens" :aria-invalid="invalid" @keydown.enter="apply" />
      <span>tokens</span>
    </div>
    <p v-if="invalid" class="reply-limit__error" role="alert">{{ REPLY_LIMIT_ERROR }}</p>
    <button v-if="dirty" class="pk-btn" type="button" :disabled="invalid" @click="apply">Apply</button>
  </div>
</template>

<style scoped>
.reply-limit { display: flex; flex-direction: column; align-items: stretch; gap: 8px; min-width: 0; }
.reply-limit__entry { position: relative; }
.reply-limit__entry :deep(input) { padding-right: 62px; max-width: none; font-variant-numeric: tabular-nums; }
.reply-limit__entry span { position: absolute; right: 10px; top: 50%; transform: translateY(-50%); pointer-events: none; color: var(--pk-text-secondary); font-size: var(--pk-font-size-sm); }
.reply-limit__error { color: var(--pk-status-warning); font-size: var(--pk-font-size-sm); }
.reply-limit > button { align-self: flex-end; }
</style>
