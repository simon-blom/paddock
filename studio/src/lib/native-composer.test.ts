import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ContentPart, Conversation } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
const fixture = vi.hoisted(() => ({ currentId: 'text', styles: {} as Record<string, string>, caps: {} as Record<string, { docParser?: boolean; sampling?: { temperature: number; source: string } }> }))
vi.mock('@/stores/models', () => ({ useModelsStore: () => ({
  ...fixture, cloudEndpoints: [],
  models: ['text', 'vision', 'speech', 'generative', 'parser'].map(id => ({ id, display: id, status: 'ok' })),
  reasoningLadderFor: () => ({ levels: ['low', 'high'], off: true, opens: 'low', preserve: false }),
  reasoningStyleFor: (id: string) => fixture.styles[id] ?? 'effort', canTranscribe: (id: string) => ['speech', 'generative'].includes(id),
  canChat: (id: string) => id !== 'speech', visionFor: (id: string) => id === 'vision' || id === 'parser',
  thinkingBudgetFor: () => false, webSearchFor: () => false,
}) }))
import { composerPresentation } from '../../native-workspace/composer'
function conversation(model: string, compareModels?: string[]): Conversation {
  return { id: 'c', model, compareModels, params: { ...DEFAULT_PARAMS }, messages: [], systemPrompt: '', connectorIds: [] } as unknown as Conversation
}
const photo: ContentPart = { type: 'image', attachmentId: 'photo', name: 'photo.png', mime: 'image/png' }
const pdf: ContentPart = { type: 'file', attachmentId: 'pdf', name: 'report.pdf', mime: 'application/pdf' }
describe('native composer capability projection', () => {
  beforeEach(() => { fixture.currentId = 'text'; fixture.caps = {}; fixture.styles = {} })
  it('keeps reasoning explanations in controls while preserving actionable attachment warnings', () => {
    fixture.styles.vision = 'toggle'
    const c = conversation('text', ['text', 'vision'])
    const plain = composerPresentation(c, [], [])
    expect(plain.warnings).toEqual([])
    expect(plain.reasoningNotice).toContain('different reasoning controls')
    const images = composerPresentation(c, [photo], [])
    expect(images.warnings).toHaveLength(1)
    expect(images.warnings[0]).toContain('cannot read images')
    expect(images.reasoningNotice).toBe(plain.reasoningNotice)
    expect(composerPresentation(conversation('text'), [], []).reasoningNotice).toBe('')
  })
  it('blocks raw images on text-only models but permits PDFs for text extraction', () => {
    expect(composerPresentation(conversation('text'), [photo], []).inputIssue).toContain('vision')
    expect(composerPresentation(conversation('text'), [pdf], []).inputIssue).toBe('')
  })
  it('permits mixed-vision compare with an explicit blind-lane warning', () => {
    const value = composerPresentation(conversation('text', ['text', 'vision']), [photo], [])
    expect(value.inputIssue).toBe('')
    expect(value.warnings[0]).toContain('text cannot read images')
  })
  it('requires audio for transcription, and a page for document parsing', () => {
    expect(composerPresentation(conversation('speech'), [], []).inputIssue).toContain('audio')
    const clip: ContentPart = { type: 'audio', attachmentId: 'a', name: 'clip.wav', mime: 'audio/wav' }
    expect(composerPresentation(conversation('speech'), [clip], []).inputIssue).toBe('')
    fixture.caps.parser = { docParser: true }
    expect(composerPresentation(conversation('parser'), [], []).inputIssue).toContain('page')
    expect(composerPresentation(conversation('parser'), [pdf], []).inputIssue).toBe('')
  })
  it('counts only selected tools, and keeps external connectors opt-in', () => {
    const groups = [{ id: 'artifacts', connectorId: '', tools: [{ name: 'create' }] }, { id: 'external', connectorId: 'external', tools: [{ name: 'search' }] }]
    const c = conversation('text')
    expect(composerPresentation(c, [], groups).toolCount).toBe(1)
    c.toolSelection = { mode: 'custom', picks: [] }
    expect(composerPresentation(c, [], groups).toolCount).toBe(0)
  })
  it('accepts generative audio, mixed speech compare and only one clip', () => {
    const clip: ContentPart = { type: 'audio', attachmentId: 'a', name: 'a.wav', mime: 'audio/wav' }
    expect(composerPresentation(conversation('generative'), [clip], [])).toMatchObject({ audioMode: false, audioOk: true, inputIssue: '' })
    expect(composerPresentation(conversation('speech', ['speech', 'generative']), [clip], [])).toMatchObject({ audioMode: true, inputIssue: '' })
    expect(composerPresentation(conversation('speech'), [clip, clip], []).inputIssue).toContain('one audio clip')
    expect(composerPresentation(conversation('speech'), [clip, pdf], []).inputIssue).toContain('only an audio clip')
    expect(composerPresentation(conversation('text', ['text', 'generative']), [clip], []).inputIssue).toContain('Every selected model')
  })
  it('projects advertised defaults without marking them as user overrides', () => {
    fixture.caps.text = { sampling: { temperature: .6, source: 'checkpoint' } }
    const value = composerPresentation(conversation('text'), [], [])
    expect(value.sampling[0]).toMatchObject({ value: .6, set: false, display: 'Default (0.6)' })
    expect(value.samplingSource).toBe('checkpoint')
    expect(value.samplerSet).toBe(false)
  })
})
