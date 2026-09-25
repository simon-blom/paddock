import type { ReadDoc, ReadRun } from './reads'

type HistoryAPI = {
  save: (doc: ReadDoc) => Promise<unknown>
  get: (id: string) => Promise<ReadDoc>
}

/** Read old browser snapshots once; SQLite is the only destination. Failed or
 * malformed snapshots remain untouched so an outage cannot destroy history. */
export async function migrateBrowserReads(storage: Storage, api: HistoryAPI): Promise<void> {
  const keys = Array.from({ length: storage.length }, (_, i) => storage.key(i))
    .filter((key): key is string => key?.startsWith('pk_reads_runs:') === true)
  const failures: string[] = []
  for (const key of keys) {
    try {
      const text = storage.getItem(key)
      if (text === null) continue
      if (text.length > 16 * 1024 * 1024) throw new Error('History exceeds the import limit')
      const data: unknown = JSON.parse(text)
      if (!Array.isArray(data) || data.length > 20) throw new Error('Invalid saved history')
      const runs: ReadRun[] = data.map((r: Partial<ReadRun>) => {
        if (!r || !Number.isFinite(r.at) || typeof r.model !== 'string' ||
          !Number.isInteger(r.port) || !r.questions || typeof r.questions !== 'object' ||
          !r.response?.answers || !Number.isFinite(r.ms)) throw new Error('Invalid saved result')
        return { ...r, state: r.state ?? '', stateMissing: r.state === undefined,
          fileName: r.fileName ?? '' } as ReadRun
      }).sort((a, b) => a.at - b.at)
      if (runs.length) {
        const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(key + '\n' + text))
        const id = 'browser-' + Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('')
        const latest = runs[runs.length - 1]!
        const doc: ReadDoc = { id, title: latest.excerpt?.slice(0, 60) || 'Imported read',
          model: latest.model, createdAt: runs[0]!.at, updatedAt: latest.at, runs }
        try { await api.save(doc) }
        catch (error) {
          // A retry or another tab may already have committed this snapshot.
          const { revision: _, ...existing } = await api.get(id).catch(() => { throw error })
          if (JSON.stringify(existing) !== JSON.stringify(doc)) throw error
        }
      }
      // Do not discard a write from an older tab that occurred during upload.
      if (storage.getItem(key) === text) storage.removeItem(key)
    } catch (error) {
      failures.push(error instanceof Error ? error.message : String(error))
    }
  }
  if (failures.length) throw new Error(`Some earlier Reads could not be imported; their browser copies were kept. ${failures[0]}`)
}
