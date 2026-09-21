import { describe, expect, it } from 'vitest'
import { withPdfViewerDocuments } from './pdf'

describe('shared viewer document lifetime', () => {
  it('serializes old-handle cleanup before the next open', async () => {
    const events: string[] = []
    let release!: () => void
    const gate = new Promise<void>(resolve => { release = resolve })
    const old = withPdfViewerDocuments(async () => {
      events.push('old starts')
      await gate
      events.push('old closes')
    })
    const next = withPdfViewerDocuments(async () => { events.push('new opens') })
    await Promise.resolve()
    expect(events).toEqual(['old starts'])
    release()
    await Promise.all([old, next])
    expect(events).toEqual(['old starts', 'old closes', 'new opens'])
  })
  it('a failed operation cannot poison the queue', async () => {
    const failed = withPdfViewerDocuments(async () => { throw new Error('Invalid PDF') })
    const next = withPdfViewerDocuments(async () => 'next document')
    await expect(failed).rejects.toThrow('Invalid PDF')
    await expect(next).resolves.toBe('next document')
  })
})
