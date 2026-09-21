import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, disposePinia, getActivePinia, setActivePinia } from 'pinia'
import { useModelsStore } from '@/stores/models'
import { useRegistryStore } from '@/stores/registry'
import type { CatalogModel } from '@/lib/api'
import { nativeTranscript } from '../../native-workspace/native-transcript'
import { DEFAULT_PARAMS, type Conversation, type Message } from '@/types/chat'

function message(id: string, text: string, parentId: string | null = null): Message {
  return { id, parentId, role: parentId ? 'assistant' : 'user', content: [{ type: 'text', text }], createdAt: 1 }
}
function conversation(messages: Message[], leafId = messages[messages.length - 1]?.id): Conversation {
  return { id: 'c', title: 'Fixture', messages, leafId, model: 'm', systemPrompt: '', params: { ...DEFAULT_PARAMS }, createdAt: 1, updatedAt: 1 }
}
describe('native transcript coverage', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', { getItem: () => null, setItem: () => {} })
    setActivePinia(createPinia())
  })
  afterEach(() => { disposePinia(getActivePinia()!); vi.unstubAllGlobals() })
  it('projects only the selected tree branch without mutating it', () => {
    const c = conversation([message('a', 'question'), message('b', 'old answer', 'a'), message('c', 'selected answer', 'a')])
    const before = JSON.stringify(c)
    expect(nativeTranscript(c).messages.map(m => m.text)).toEqual(['question', 'selected answer'])
    expect(JSON.stringify(c)).toBe(before)
  })
  it('preserves reasoning, final syntax, errors and cancellation state', () => {
    const m = { ...message('a', '```swift\nlet x = 1\n```'), reasoning: 'Thinking', stopped: true, error: 'error' }
    expect(nativeTranscript(conversation([m])).messages[0]).toMatchObject({ text: m.content[0].type === 'text' ? m.content[0].text : '', reasoning: 'Thinking', stopped: true, error: 'error' })
  })
  it('keeps tools and general attachments in native presentation', () => {
    for (const extra of [{ content: [{ type: 'file', attachmentId: 'file', name: 'data.zip', mime: 'application/zip' }] }, { toolCalls: [{ id: 'tool', name: 'read', serverLabel: 'test', status: 'pending', arguments: '{}', approvalId: 'approval' }] }]) {
      const c = conversation([{ ...message('a', 'question'), ...extra } as Message])
      expect(nativeTranscript(c).available).toBe(true)
      expect(nativeTranscript(c).messages).toHaveLength(1)
    }
  })
  it('keeps whole compare groups, independent streams and one anchor branch switch', () => {
    const q = message('q', 'Compare these')
    const lanes = (group: string, suffix: string) => [
      { ...message(`local${suffix}`, '## Local\n\n```swift\nlet x = 1\n```', 'q'), model: 'qwen', group, streaming: true },
      { ...message(`cloud${suffix}`, '## Cloud\n\nDifferent answer', 'q'), model: 'cloud:account:meta/muse@meta', group, stopped: true, incomplete: 'length' as const },
    ]
    const c = conversation([q, ...lanes('old', '0'), ...lanes('new', '1')], 'cloud1')
    c.compareModels = ['qwen', 'cloud:account:meta/muse@meta']
    const before = JSON.stringify(c)
    const result = nativeTranscript(c)
    expect(result.available).toBe(true)
    expect(result.messages.map(m => [m.id, m.group])).toEqual([['q', null], ['local1', 'local1'], ['cloud1', 'local1']])
    expect(result.messages[1]).toMatchObject({ streaming: true, actions: { retry: false, continueReply: false, branch: { index: 2, count: 2, previous: 'local0' } } })
    expect(result.messages[2]).toMatchObject({ stopped: true, incomplete: true, model: 'cloud:account:meta/muse@meta', actions: { retry: false, continueReply: false, branch: null } })
    expect(result.messages[0].actions?.edit).toBe(true)
    expect(JSON.stringify(c)).toBe(before)
    // A completed lane must not change identity when its sibling receives a delta.
    c.messages.find(m => m.id === 'local1')!.content = [{ type: 'text', text: 'More tokens' }]
    expect(nativeTranscript(c).messages[2]).toEqual(result.messages[2])
    c.leafId = 'local0'
    expect(nativeTranscript(c).messages.map(m => m.id)).toEqual(['q', 'local0', 'cloud0'])
  })
  it('projects every stored document badge without bytes and keeps native text', () => {
    const c = conversation([{ ...message('u', 'Read these'), content: [
      { type: 'text', text: 'Read these' },
      { type: 'file', attachmentId: 'pdf1', name: 'one.pdf', mime: 'application/pdf', pages: 6, pageRange: '2-4', pdfMode: 'text' },
      { type: 'file', attachmentId: 'pdf2', name: 'two.pdf', mime: 'application/pdf' },
      { type: 'file', attachmentId: 'docx', name: 'three.docx', mime: '' },
      { type: 'image', attachmentId: 'image', name: 'four.png', mime: 'image/png', thumbUrl: 'PRIVATE_PIXELS' },
    ] }])
    const result = nativeTranscript(c)
    expect(result.available).toBe(true)
    expect(result.messages[0].attachments.map(a => a.id)).toEqual(['pdf1', 'pdf2', 'docx', 'image'])
    expect(result.messages[0].attachments[0]).toMatchObject({ kind: 'pdf', pages: 6, pageRange: '2-4', textOnly: true })
    expect(JSON.stringify(result)).not.toContain('PRIVATE_PIXELS')
    c.messages[0].docRun = {} as never
    expect(nativeTranscript(c).available).toBe(true)
  })
  it('keeps large conversations native without truncating the original', () => {
    const c = conversation([message('a', '🦄'.repeat(40000))])
    expect(nativeTranscript(c).available).toBe(true)
    expect(nativeTranscript(c).messages[0].text).toBe('🦄'.repeat(40000))
    expect(c.messages[0].content).toEqual([{ type: 'text', text: '🦄'.repeat(40000) }])
    expect(nativeTranscript(null)).toEqual({ available: true, notice: '', conversationId: null, leafId: null, messages: [] })
  })
  it('keeps stopped-model author identity and recorded speculation, not the current composer model', () => {
    useModelsStore().currentId = 'different-model'
    // Recorded off must hide the badge, not inherit a currently active mode.
    useModelsStore().models = [{ id: 'qwen-fixture', spec: 'MTP+DFlash2', kind: 'chat', status: 'ok', ownedBy: 'test' }]
    useRegistryStore().models = [{ id: 'qwen-fixture', display: 'Qwen 3.8 27B', vendor: 'Alibaba', artifacts: [] } as unknown as CatalogModel]
    const answer: Message = { ...message('a', 'A reply', 'q'),
      run: { model: 'qwen-fixture', spec: 'off', params: { ...DEFAULT_PARAMS }, tools: [], at: 1 },
      usage: { promptTokens: 24, completionTokens: 120, tps: 60, ms: 5600, ttftMs: 350 } }
    const result = nativeTranscript(conversation([message('q', 'Question'), answer]))
    expect(result.messages[1]).toMatchObject({ model: 'qwen-fixture', chrome: {
      // Legacy stored tps has no decode-duration provenance. The footer must
      // derive the honest end-to-end rate (120 / 5.6), not repeat that number.
      modelName: 'Qwen 3.8 27B', vendor: 'Alibaba', spec: '', footer: '120 tokens · 21 tok/s · 5.6s',
    } })
    expect(result.messages[0].chrome).toBeNull()
    expect(result.messages[1].chrome?.sections.flatMap(s => s.rows)).toContainEqual({ label: 'Speculation', value: 'off' })
    expect(answer.run?.spec).toBe('off')
    expect(JSON.stringify(result)).not.toContain('different-model')
  })
  it('retains cloud maker identity after the endpoint disappears and old turns without usage', () => {
    const answer = { ...message('a', 'Reply', 'q'), model: 'cloud:removed-endpoint:anthropic/claude-fixture@host' }
    const result = nativeTranscript(conversation([message('q', 'Question'), answer])).messages[1]
    expect(result.chrome?.vendor).toBe('Anthropic')
    expect(result.chrome?.modelName).not.toContain('removed-endpoint')
    expect(result.chrome?.footer).toBe('')
    expect(result.chrome?.sections).toEqual([])
  })
  it('includes metadata in the byte budget and never spreads raw run fields', () => {
    const answer = { ...message('a', 'Reply', 'q'), run: { model: 'fixture', params: { ...DEFAULT_PARAMS }, tools: [], at: 1, authorization: 'NEVER_PROJECT', systemPrompt: 'Short prompt' } }
    expect(JSON.stringify(nativeTranscript(conversation([message('q', 'Question'), answer])))).not.toContain('NEVER_PROJECT')
    answer.run.systemPrompt = '🦄'.repeat(40000)
    expect(nativeTranscript(conversation([message('q', 'Question'), answer])).available).toBe(true)
  })
})
