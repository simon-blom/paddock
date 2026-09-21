import { describe, expect, it } from 'vitest'
import { attachmentChoices, parseCommand, storedAttachment } from '../../native-workspace/protocol'

describe('native content command boundary', () => {
  const command = { version: 1, id: 'request-1', kind: 'send', payload: { text: 'Hello' } }
  it('accepts only the versioned command vocabulary', () => {
    expect(parseCommand(command)).toEqual(command)
    for (const value of [{ ...command, version: 2 }, { ...command, kind: 'readFile' }, { ...command, script: 'alert(1)' }, { ...command, id: '../file' }, { ...command, payload: [] }]) expect(() => parseCommand(value)).toThrow()
  })
  it('attachment metadata has no path and a strict size limit', () => {
    expect(storedAttachment({ id: 'attachment-1', name: 'report.pdf', mime: 'application/pdf', size: 100 })).toEqual({ id: 'attachment-1', name: 'report.pdf', mime: 'application/pdf', size: 100 })
    for (const size of [-1, NaN, Infinity, 100 * 1024 * 1024 + 1]) expect(() => storedAttachment({ id: 'a', name: 'a', mime: '', size })).toThrow()
    expect(() => storedAttachment({ id: '../secret', name: 'a', mime: '', size: 1 })).toThrow()
    expect(() => storedAttachment({ id: 'a', name: 'a', mime: '', size: 1, path: '/private/file' })).toThrow()
  })
  it('rejects duplicate attachments and backwards, fractional or negative page ranges', () => {
    expect(attachmentChoices([{ id: 'a', from: 2, to: 4, text: true }])).toEqual([{ id: 'a', from: 2, to: 4, text: true, detail: undefined }])
    for (const value of [[{ id: 'a' }, { id: 'a' }], [{ id: 'a', from: 4, to: 2 }], [{ id: 'a', from: 1.5 }], [{ id: 'a', to: 0 }], [{ id: 'a', detail: 'original' }]]) expect(() => attachmentChoices(value)).toThrow()
  })
})
