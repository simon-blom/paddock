import { defineStore } from 'pinia'
import { ref } from 'vue'
import { uuid } from '@/lib/uuid'
import { readHistoryApi, readsApi, type SavedReadSet } from '@/lib/api'
import type { ReadDoc, ReadSummary } from '@/lib/reads'

/** The Reads page's two kinds of record, both kept by the manager.
 *
 *  Saved read SETS are named question sets (/api/reads) on the Prompts
 *  pattern: the same optimistic revision on save and delete, the acknowledged
 *  row published rather than a refresh that might fail after commit.
 *
 *  READS are the page's history (/api/read-history): a text, its questions
 *  and every run, listed in the side panel the way a chat lists
 *  conversations. The list holds summaries; a read is fetched whole to open
 *  it and saved whole after each run, never from a summary. */
export const useReadsStore = defineStore('reads', () => {
  /** The running reader the page sends to, chosen in the header's model
   *  picker while the Reads page is open - the one place a model is picked,
   *  as on the chat page. 0 until a reader is running. */
  const readerPort = ref(0)
  const sets = ref<SavedReadSet[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  let refreshGeneration = 0

  async function refresh(): Promise<void> {
    const generation = ++refreshGeneration
    loading.value = true
    error.value = null
    try {
      const rows = await readsApi.list()
      if (generation === refreshGeneration) sets.value = rows
    } catch (e) {
      if (generation === refreshGeneration) error.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (generation === refreshGeneration) loading.value = false
    }
  }

  /** Create or update a named set; returns the saved record. */
  async function save(name: string, body: string, id?: string, revision?: string): Promise<SavedReadSet> {
    if (!name.trim() || !body.trim()) throw new Error('Enter a name for the set')
    const rec: SavedReadSet = {
      id: id ?? uuid(),
      name: name.trim(),
      body,
      revision: revision ?? (id ? sets.value.find((s) => s.id === id)?.revision : ''),
    }
    const reply = await readsApi.save(rec)
    ++refreshGeneration
    loading.value = false
    error.value = null
    const saved = reply.set ?? rec
    sets.value = [saved, ...sets.value.filter((s) => s.id !== saved.id)]
    return saved
  }

  async function remove(id: string, revision?: string): Promise<void> {
    await readsApi.remove(id, revision ?? sets.value.find((s) => s.id === id)?.revision)
    ++refreshGeneration
    loading.value = false
    error.value = null
    sets.value = sets.value.filter((s) => s.id !== id)
  }

  // ── reads: the side panel's history ─────────────────────────────────────

  const reads = ref<ReadSummary[]>([])
  const readsLoaded = ref(false)
  const readsError = ref<string | null>(null)
  let readsGeneration = 0

  async function refreshReads(): Promise<void> {
    const generation = ++readsGeneration
    try {
      const rows = await readHistoryApi.list()
      if (generation === readsGeneration) {
        reads.value = rows
        readsError.value = null
      }
    } catch (e) {
      if (generation === readsGeneration) readsError.value = e instanceof Error ? e.message : String(e)
    } finally {
      if (generation === readsGeneration) readsLoaded.value = true
    }
  }

  function loadRead(id: string): Promise<ReadDoc> {
    return readHistoryApi.get(id)
  }

  /** Save a whole read and move its row to the top: the list follows what
   *  the manager acknowledged, not a refetch that could race the next run. */
  async function saveRead(doc: ReadDoc): Promise<ReadSummary> {
    const reply = await readHistoryApi.save(doc)
    doc.revision = reply.read?.revision
    ++readsGeneration
    const row: ReadSummary = reply.read ?? {
      id: doc.id,
      title: doc.title,
      model: doc.model,
      runs: doc.runs.length,
      createdAt: doc.createdAt,
      updatedAt: doc.updatedAt,
    }
    reads.value = [row, ...reads.value.filter((r) => r.id !== row.id)]
    readsError.value = null
    return row
  }

  async function removeRead(id: string): Promise<void> {
    const doc = await readHistoryApi.get(id)
    await readHistoryApi.remove(id, doc.revision!)
    ++readsGeneration
    reads.value = reads.value.filter((r) => r.id !== id)
  }

  /** Rename from the side panel: the read is fetched whole and saved whole,
   *  so a title change never travels on a summary. */
  async function renameRead(id: string, title: string): Promise<ReadDoc> {
    const doc = await readHistoryApi.get(id)
    const next = { ...doc, title: title.trim() || doc.title }
    const reply = await readHistoryApi.save(next)
    next.revision = reply.read?.revision
    ++readsGeneration
    // a rename keeps the row where it is: position follows the last run
    reads.value = reads.value.map((r) => (r.id === id ? (reply.read ?? { ...r, title: next.title }) : r))
    return next
  }

  return {
    readerPort,
    sets,
    loading,
    error,
    refresh,
    save,
    remove,
    reads,
    readsLoaded,
    readsError,
    refreshReads,
    loadRead,
    saveRead,
    removeRead,
    renameRead,
  }
})
