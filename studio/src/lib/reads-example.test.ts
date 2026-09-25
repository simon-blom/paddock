import { describe, expect, it } from 'vitest'
import fixture from '../../../apps/macos/Sources/PaddockConversationCore/Resources/reads-example.json'
import { readExample } from './reads-example'
import { deriveId, requestBody, toWire } from './reads'

describe('shared native / web Reads example', () => {
  it('keeps the complete request and all three question types in authored order', () => {
    const example = readExample()
    expect({ ...example, questions: toWire(example.questions) }).toEqual(fixture)
    expect(example.questions.map((q) => [q.id, q.type])).toEqual([
      ['need_action_within', 'noul'],
      ['message_about', 'choice'],
      ['upset_sender', 'score'],
    ])
    expect(example.questions[1].options.map((o) => o.name)).toEqual([
      'outage', 'billing', 'feature', 'other',
    ])
    expect(example.questions[2].levels).toEqual(['calm', 'annoyed', 'furious'])
    expect(requestBody(example.state, example.questions, example.samples)).toEqual({
      state: fixture.state,
      questions: fixture.questions,
    }) // Omitted samples and explicit "auto" both request adaptive sampling.
  })

  it('keeps example IDs descriptive and editable, like newly authored questions', () => {
    const example = readExample()
    example.questions.forEach((q, i) => {
      expect(q.id).toBe(deriveId(q.instructions, example.questions.slice(0, i).map((p) => p.id)))
      expect(q.idTouched).toBe(false)
    })
  })

  it('never shares mutable editor state or row identities between drafts', () => {
    const first = readExample()
    const second = readExample()
    expect(
      first.questions.some((q) => second.questions.some((other) => other.key === q.key)),
    ).toBe(false)
    first.state = 'Changed'
    first.questions[0].yesMeans = 'Changed'
    first.questions[1].options[0].name = 'Changed'
    first.questions[2].levels[0] = 'Changed'
    expect({ ...second, questions: toWire(second.questions) }).toEqual(fixture)
    expect(toWire(readExample().questions)).toEqual(fixture.questions)
  })
})
