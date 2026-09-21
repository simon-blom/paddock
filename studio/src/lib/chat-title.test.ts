import { describe, expect, it, vi } from 'vitest'
import { cleanTitle, titleTranscript, TitleGenerator } from './chat-title'
import { DEFAULT_PARAMS, type Conversation } from '@/types/chat'

const target = { model: 'fixture', endpoint: '/api/runners/16555/v1/responses' }
const answer = (text = 'Planning a native studio') => Response.json({ status: 'completed', output: [
  { type: 'reasoning', summary: [{ text: 'Never use reasoning as a label' }] },
  { type: 'message', content: [{ type: 'output_text', text }] },
] })
describe('bounded conversation title requests', () => {
  it('never admits background title work while a foreground response is running', async () => {
    const generator = new TitleGenerator(), fetcher = vi.fn(async () => answer())
    generator.setForeground(true)
    await expect(generator.generate('c', 'x', target, fetcher)).rejects.toThrow('active responses')
    expect(fetcher).not.toHaveBeenCalled()
    generator.setForeground(false)
    await generator.generate('c', 'x', target, fetcher)
    expect(fetcher).toHaveBeenCalledOnce()
  })
  it('uses only active text and attachment names, not hidden branches or privileged context', () => {
    const c: Conversation = { id: 'c', title: 'New chat', model: 'fixture', params: { ...DEFAULT_PARAMS }, systemPrompt: 'SECRET SYSTEM', createdAt: 0, updatedAt: 0, leafId: 'a', messages: [
      { id: 'u', parentId: null, role: 'user', createdAt: 0, content: [{ type: 'text', text: 'Help with my report' }, { type: 'image', name: 'chart.png', mime: 'image/png', attachmentId: 'image', modelUrl: 'SECRET BYTES' }] },
      { id: 'hidden', parentId: 'u', role: 'assistant', createdAt: 1, content: [{ type: 'text', text: 'HIDDEN BRANCH' }] },
      { id: 'a', parentId: 'u', role: 'assistant', createdAt: 1, reasoning: 'SECRET THOUGHT', content: [{ type: 'text', text: 'Review the annual report' }] },
    ] }
    const text = titleTranscript(c)
    expect(text).toContain('chart.png')
    expect(text).toContain('annual report')
    expect(text).not.toMatch(/SECRET|HIDDEN/)
    c.messages[0].content = [{ type: 'text', text: 'x'.repeat(100_000) }]
    expect(titleTranscript(c).length).toBeLessThanOrEqual(5000)
  })
  it('keeps the request stateless, bounded, tool-free and on the selected relay', async () => {
    const fetcher = vi.fn(async () => answer())
    expect(await new TitleGenerator().generate('c', 'user: test', target, fetcher)).toEqual({ title: 'Planning a native studio', cost: undefined })
    const [url, options] = fetcher.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toBe(target.endpoint)
    const request = JSON.parse(String(options.body))
    expect(request).toMatchObject({ model: 'fixture', store: false, stream: false, max_output_tokens: 128 })
    expect(request.tools).toBeUndefined()
    expect(request.previous_response_id).toBeUndefined()
  })
  it('rejects incomplete, empty, multiline, markup and oversize results', async () => {
    for (const value of ['', 'hello\nworld', '<think>no</think>', '`code`', 'x'.repeat(101)]) expect(() => cleanTitle(value)).toThrow()
    expect(cleanTitle('Title: “A clear label”')).toBe('A clear label')
    await expect(new TitleGenerator().generate('c', 'x', target, async () => Response.json({ status: 'incomplete', output: [] }))).rejects.toThrow('did not finish')
    await expect(new TitleGenerator().generate('c', 'x', target, async () => new Response('x'.repeat(65_537)))).rejects.toThrow('size limit')
    await expect(new TitleGenerator().generate('c', 'x', target, async () => new Response('', { status: 503 }))).rejects.toThrow('503')
  })
  it('discards late output even when the transport ignores cancellation', async () => {
    const generator = new TitleGenerator()
    let finish!: (response: Response) => void
    const running = generator.generate('c', 'x', target, () => new Promise(resolve => { finish = resolve }))
    generator.cancel('other')
    generator.cancel('c')
    finish(answer())
    await expect(running).rejects.toMatchObject({ name: 'AbortError' })
  })
  it('new work preempts an older title', async () => {
    const generator = new TitleGenerator()
    let finish!: (response: Response) => void
    const first = generator.generate('a', 'x', target, () => new Promise(resolve => { finish = resolve }))
    const second = await generator.generate('b', 'y', target, async () => answer('The new title'))
    finish(answer('An obsolete title'))
    await expect(first).rejects.toMatchObject({ name: 'AbortError' })
    expect(second.title).toBe('The new title')
  })
})
