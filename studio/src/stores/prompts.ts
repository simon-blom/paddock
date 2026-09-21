import { defineStore } from 'pinia'
import { uuid } from '@/lib/uuid'
import { ref } from 'vue'
import { promptsApi, type SavedPrompt } from '@/lib/api'

function uid(): string {
  return uuid()
}

/** The reusable system-prompt library, backed by the server store (/api/prompts).
 *  Separate from a conversation's own `systemPrompt`: these are named, saved
 *  prompts the user can apply to any chat. */
export const usePromptsStore = defineStore('prompts', () => {
  const prompts = ref<SavedPrompt[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  let refreshGeneration = 0

  async function refresh(): Promise<void> {
    const generation = ++refreshGeneration
    loading.value = true
    error.value = null
    try {
      const rows = await promptsApi.list()
      if (generation === refreshGeneration) prompts.value = rows
    } catch (e) {
      if (generation === refreshGeneration) error.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (generation === refreshGeneration) loading.value = false
    }
  }

  /** Create or update a named prompt; returns the saved record. */
  async function save(name: string, body: string, id?: string, revision?: string): Promise<SavedPrompt> {
    if (!name.trim() || !body.trim()) throw new Error('Enter a name and prompt text')
    const rec: SavedPrompt = { id: id ?? uid(), name: name.trim(), body,
      revision: revision ?? (id ? prompts.value.find(p => p.id === id)?.revision : '') }
    const reply = await promptsApi.save(rec)
    // Publish the acknowledged row, not a refresh that might fail after commit.
    // Also invalidate a list request started before this mutation.
    ++refreshGeneration; loading.value = false; error.value = null
    const saved = reply.prompt ?? rec // Compatibility with older web Managers.
    prompts.value = [saved, ...prompts.value.filter(p => p.id !== saved.id)]
    return saved
  }

  async function remove(id: string, revision?: string): Promise<void> {
    await promptsApi.remove(id, revision ?? prompts.value.find(p => p.id === id)?.revision)
    ++refreshGeneration; loading.value = false; error.value = null
    prompts.value = prompts.value.filter((p) => p.id !== id)
  }

  return { prompts, loading, error, refresh, save, remove }
})
