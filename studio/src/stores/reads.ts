import { defineStore } from 'pinia'
import { ref } from 'vue'
import { uuid } from '@/lib/uuid'
import { readsApi, type SavedReadSet } from '@/lib/api'
import type { ReadRun } from '@/lib/reads'

/** Saved read sets - named question sets for the Reads page, backed by the
 *  server store (/api/reads) on the Prompts pattern: the same optimistic
 *  revision on save and delete, the acknowledged row published rather than
 *  a refresh that might fail after commit.
 *
 *  Run history is LOCAL and per set: the last few reads (excerpt, answers,
 *  model, time) so a tweak can be read against the previous result. It is a
 *  convenience, so it lives in localStorage and the page renders without it. */
export const useReadsStore = defineStore('reads', () => {
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
    forgetRuns(id)
  }

  // ── local run history ───────────────────────────────────────────────────

  const RUNS_KEEP = 10
  function runsKey(setId: string | undefined): string {
    return `pk_reads_runs:${setId ?? 'draft'}`
  }
  function runsOf(setId: string | undefined): ReadRun[] {
    try {
      const raw = localStorage.getItem(runsKey(setId))
      if (!raw) return []
      const v = JSON.parse(raw) as unknown
      return Array.isArray(v) ? (v as ReadRun[]) : []
    } catch {
      return []
    }
  }
  /** Keep a run; the newest first, the oldest dropped past RUNS_KEEP. A full
   *  storage never fails the read - the run is simply not remembered. */
  function recordRun(setId: string | undefined, run: ReadRun): ReadRun[] {
    const next = [run, ...runsOf(setId)].slice(0, RUNS_KEEP)
    try {
      localStorage.setItem(runsKey(setId), JSON.stringify(next))
    } catch {
      /* quota or private mode: history is a convenience */
    }
    return next
  }
  function forgetRuns(setId: string | undefined): void {
    try {
      localStorage.removeItem(runsKey(setId))
    } catch {
      /* nothing to forget */
    }
  }

  return { sets, loading, error, refresh, save, remove, runsOf, recordRun, forgetRuns }
})
