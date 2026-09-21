import { describe, expect, it } from 'vitest'
import { attachmentChoices } from '../../native-workspace/protocol'
import { selectedAttachment } from '../../native-workspace/attachment-selection'
import type { FilePart } from '@/types/chat'

const pdf: FilePart = { type: 'file', attachmentId: 'pdf', name: 'report.pdf', mime: 'application/pdf', size: 1000, pages: 12 }
function select(options: object, file = pdf) {
  return selectedAttachment(file, attachmentChoices([{ id: file.attachmentId, ...options }])[0])
}
describe('native PDF selections use shared attachment semantics', () => {
  it('preserves originals and sends inclusive, single and open ranges', () => {
    for (const [options, range] of [
      [{ from: 2, to: 4 }, '2-4'], [{ from: 3, to: 3 }, '3'],
      [{ from: 5 }, '5-'], [{ to: 4 }, '1-4'], [{}, undefined],
    ] as const) {
      expect(select(options)).toEqual({ ...pdf, pageRange: range, pdfMode: undefined })
    }
    expect(pdf.pageRange).toBeUndefined()
    expect(pdf.pdfMode).toBeUndefined()
  })
  it('text-only is per file and resetting restores automatic/all pages', () => {
    const text = select({ text: true, from: 2, to: 4 }) as FilePart
    expect(text.pdfMode).toBe('text')
    expect(text.pageRange).toBe('2-4')
    expect(select({}, text)).toEqual({ ...pdf, pageRange: undefined, pdfMode: undefined })
    expect(select({}, { ...pdf, attachmentId: 'second' }).type).toBe('file')
    expect(pdf.pdfMode).toBeUndefined()
  })
  it('rejects invalid selections before a turn can be admitted', () => {
    for (const options of [{ from: 13 }, { to: 13 }, { from: 4, to: 2 }, { from: 0 }, { to: 1.2 }, { from: Number.MAX_SAFE_INTEGER + 1 }]) {
      expect(() => select(options)).toThrow()
    }
    expect(() => selectedAttachment(undefined, { id: 'pdf' })).toThrow('not ready')
    expect(() => selectedAttachment(pdf, { id: 'wrong' })).toThrow('not ready')
    expect(() => select({ from: 2 }, { ...pdf, name: 'notes.txt', mime: 'text/plain' })).toThrow('only available')
    expect(() => select({ text: true }, { ...pdf, name: 'notes.txt', mime: 'text/plain' })).toThrow('only available')
  })
  it('supports unknown counts and MIME-only PDF detection without inventing a total', () => {
    expect(select({ from: 2 }, { ...pdf, name: 'document', pages: undefined })).toMatchObject({ pageRange: '2-', pages: undefined })
    expect(selectedAttachment({ type: 'image', attachmentId: 'tiff', mime: 'image/tiff', name: 'scan.tif' }, { id: 'tiff', from: 2, to: 4, detail: 'high' })).toMatchObject({ pageRange: '2-4', detail: 'high' })
  })
})
