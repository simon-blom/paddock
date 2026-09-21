import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { createPinia, disposePinia, getActivePinia, setActivePinia } from 'pinia'
import { nativeTranscript } from '../../native-workspace/native-transcript'
import { exportTranscript } from '../../native-workspace/speech'
import { DEFAULT_PARAMS, type Conversation, type Message } from '@/types/chat'

function fixture(): Conversation {
  const user: Message = { id: 'u', parentId: null, role: 'user', createdAt: 1, content: [
    { type: 'audio', attachmentId: 'clip_1', name: 'Recording.m4a', mime: 'audio/mp4', size: 100, durationS: 3 },
  ] }
  const answer = (id: string, text: string): Message => ({ id, parentId: 'u', role: 'assistant', group: 'g', createdAt: 2,
    content: [{ type: 'text', text }], transcript: { language: 'sv', durationS: 3,
      segments: [{ start: 0, end: 3, text }], words: text.split(' ').map((word, i) => ({ word, start: i, end: i + 0.5, confidence: 0.3 })),
    } })
  return { id: 'c', title: 'Audio', messages: [user, answer('a', 'Hej världen'), answer('b', 'Hej Paddock')], leafId: 'b',
    model: 'speech', systemPrompt: '', params: { ...DEFAULT_PARAMS }, createdAt: 1, updatedAt: 2 }
}
describe('native speech projection', () => {
  beforeEach(() => { vi.stubGlobal('localStorage', { getItem: () => null, setItem() {} }); setActivePinia(createPinia()) })
  afterEach(() => { disposePinia(getActivePinia()!); vi.unstubAllGlobals() })
  it('keeps audio comparison native with one original, shared timing and symmetric differences', () => {
    const c = fixture(), before = JSON.stringify(c), result = nativeTranscript(c)
    expect(result.available).toBe(true)
    expect(result.messages[0].audioClips).toEqual([{ id: 'clip_1', name: 'Recording.m4a', mime: 'audio/mp4', size: 100, duration: 3 }])
    for (const m of result.messages.slice(1)) {
      expect(m.audioClips).toEqual([])
      expect(m.speech?.clip?.id).toBe('clip_1')
      expect(m.speech?.differs).toEqual([1])
      expect(m.speech?.words[0]).toMatchObject({ word: 'Hej', start: 0, end: 0.5, confidence: 0.3 })
      expect(m.speech?.subtitleExport).toBe(true)
    }
    expect(JSON.stringify(c)).toBe(before)
  })
  it('does not invent word timing for text-only speech or sentence-only models', () => {
    const c = fixture(), m = c.messages[2]
    m.transcript = {}
    const result = nativeTranscript(c)
    expect(result.available).toBe(true)
    expect(result.messages[2].speech?.words[0].start).toBeUndefined()
    expect(result.messages[2].speech?.subtitleExport).toBe(false)
    m.transcript = { segments: [{ start: 1, end: 3, text: 'Hej Paddock' }] }
    expect(nativeTranscript(c).messages[2].speech?.words[0]).toMatchObject({ start: 1 })
    expect(nativeTranscript(c).messages[2].speech?.words[0].end).toBeUndefined()
  })
  it('exports the stored transcript and rejects stale branches, streams and invented subtitle times', () => {
    const c = fixture(), p = { conversationId: 'c', leafId: 'b', messageId: 'a', format: 'srt' }
    expect(exportTranscript(c, p).text).toContain('Hej världen')
    expect(exportTranscript(c, { ...p, format: 'txt' }).text).toBe('Hej världen')
    expect(() => exportTranscript(c, { ...p, leafId: 'old' })).toThrow('branch changed')
    c.messages[1].streaming = true
    expect(() => exportTranscript(c, p)).toThrow('completed')
    c.messages[1].streaming = false; c.messages[1].transcript = {}
    expect(() => exportTranscript(c, p)).toThrow('subtitle times')
  })
})
