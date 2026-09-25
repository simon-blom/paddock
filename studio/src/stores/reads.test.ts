import { beforeEach, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useReadsStore } from './reads'
import { readHistoryApi } from '@/lib/api'
import type { ReadDoc } from '@/lib/reads'

vi.mock('@/lib/api', () => ({
  readsApi: {},
  readHistoryApi: { list: vi.fn(), get: vi.fn(), save: vi.fn(), remove: vi.fn() },
}))
beforeEach(() => { setActivePinia(createPinia()); vi.resetAllMocks() })
const doc = (): ReadDoc => ({ id: 'read', title: 'Ticket', model: 'diffusion', createdAt: 1, updatedAt: 2, runs: [] })
const row = { id: 'read', title: 'Ticket', model: 'diffusion', createdAt: 1, updatedAt: 2, runs: 0 }

it('keeps the acknowledged revision for the next run without refetching the full document', async () => {
  const store = useReadsStore()
  const d = doc()
  vi.mocked(readHistoryApi.save).mockResolvedValue({ ok: true, read: { ...row, revision: 'r1' } })
  await store.saveRead(d)
  expect(d.revision).toBe('r1')
  expect(store.reads).toHaveLength(1)
  expect(readHistoryApi.get).not.toHaveBeenCalled()
})

it('does not publish a successful save or drop the current result on a conflict', async () => {
  const store = useReadsStore()
  const d = { ...doc(), revision: 'reviewed' }
  vi.mocked(readHistoryApi.save).mockRejectedValue(Error('changed'))
  await expect(store.saveRead(d)).rejects.toThrow('changed')
  expect(d.revision).toBe('reviewed')
  expect(store.reads).toEqual([])
})

it('deletes only the document revision read from SQLite', async () => {
  const store = useReadsStore()
  vi.mocked(readHistoryApi.get).mockResolvedValue({ ...doc(), revision: 'r2' })
  vi.mocked(readHistoryApi.remove).mockResolvedValue(undefined)
  await store.removeRead('read')
  expect(readHistoryApi.remove).toHaveBeenCalledWith('read', 'r2')
})
