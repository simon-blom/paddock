import { describe, expect, it } from 'vitest'
import { migrateBrowserReads } from './reads-migration'
import type { ReadDoc } from './reads'

function storage() {
  const rows = new Map<string, string>()
  return { get length() { return rows.size }, key: (i: number) => [...rows.keys()][i] ?? null,
    getItem: (k: string) => rows.get(k) ?? null, setItem: (k: string, v: string) => { rows.set(k, v) },
    removeItem: (k: string) => { rows.delete(k) }, clear: () => rows.clear() } satisfies Storage
}
const key = 'pk_reads_runs:draft'
const old = JSON.stringify([{ at: 1234, model: 'diffusion', port: 11543, excerpt: 'Part of input', chars: 1000,
  questions: { z: { type: 'noul' } }, samples: 'auto', response: { answers: { z: {} } }, ms: 1.5 }])

describe('browser Reads migration to SQLite', () => {
  it('removes only acknowledged snapshots and never substitutes an excerpt for full input', async () => {
    const s = storage(); s.setItem(key, old); s.setItem('unrelated', 'keep')
    let saved: ReadDoc | undefined
    await migrateBrowserReads(s, { save: async (d) => { saved = d }, get: async () => { throw Error() } })
    expect(saved?.runs[0]?.state).toBe('')
    expect(saved?.runs[0]?.stateMissing).toBe(true)
    expect(saved?.runs[0]?.chars).toBe(1000)
    expect(s.getItem(key)).toBeNull()
    expect(s.getItem('unrelated')).toBe('keep')
  })
  it('retains history when offline or malformed', async () => {
    const s = storage(); s.setItem(key, old)
    const offline = { save: async () => { throw Error('offline') }, get: async () => { throw Error('offline') } }
    await expect(migrateBrowserReads(s, offline)).rejects.toThrow('kept')
    expect(s.getItem(key)).toBe(old)
    s.setItem(key, '{broken')
    await expect(migrateBrowserReads(s, offline)).rejects.toThrow('kept')
    expect(s.getItem(key)).toBe('{broken')
  })
  it('keeps the source when browser storage refuses removal', async () => {
    const s = storage(); s.setItem(key, old)
    s.removeItem = () => { throw Error('blocked') }
    await expect(migrateBrowserReads(s, { save: async () => {}, get: async () => { throw Error() } })).rejects.toThrow('kept')
    expect(s.getItem(key)).toBe(old)
  })
  it('retries idempotently after the server committed but the reply was lost', async () => {
    const s = storage(); s.setItem(key, old)
    let committed: ReadDoc | undefined
    await migrateBrowserReads(s, {
      save: async (d) => { committed = d; throw Error('lost reply') },
      get: async (id) => { expect(id).toBe(committed!.id); return { ...committed!, revision: 'r' } },
    })
    expect(s.getItem(key)).toBeNull()
  })
  it('does not remove a newer write by an older tab during migration', async () => {
    const s = storage(); s.setItem(key, old)
    await migrateBrowserReads(s, { save: async () => { s.setItem(key, 'newer') }, get: async () => { throw Error() } })
    expect(s.getItem(key)).toBe('newer')
  })
})
