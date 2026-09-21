// The reply budget of a cloud lane, anchored on the last count a server gave.
//
// A cap derived from a prompt guessed end to end (4 chars a token, tool schemas
// invisible) needed a flat 1024-token margin and an unconditional 512 floor to
// survive its own error. `replyPrompt` replaces the guess with the previous
// turn's `usage.promptTokens` wherever that number is known to describe the
// same prefix, and `windowRemaining` then charges its margin to the remainder
// only. These tests pin when the anchor is trusted, when it is not, and what
// the arithmetic does at the edges. The shared web/native rows are in
// native-reply-budget-parity.test.ts.

import { describe, expect, it } from 'vitest'
import type { Conversation, Message, RunMeta } from '@/types/chat'
import { DEFAULT_PARAMS } from '@/types/chat'
import {
  MIN_USEFUL_REPLY,
  parseContextOverflow,
  promptShape,
  promptTokensFrom,
  replyPrompt,
  windowRemaining,
} from './tokens'

// -- fixtures ----------------------------------------------------------------
//
// Round numbers, so every expectation is arithmetic that can be redone by
// hand: 4 chars a token and 4 tokens of per-message overhead.

const MODEL = 'cloud/test'
const TOOLS = 'github|search#c'
/** 800 chars = 200 tokens, + 4 overhead. */
const TEXT = 't'.repeat(800)
const MSG_TOKENS = 204

function msg(i: number, extra: Partial<Message> = {}): Message {
  return {
    id: `m${i}`,
    parentId: i === 0 ? null : `m${i - 1}`,
    role: i % 2 === 0 ? 'user' : 'assistant',
    content: [{ type: 'text', text: TEXT }],
    createdAt: i,
    ...extra,
  }
}

function run(extra: Partial<RunMeta> = {}): RunMeta {
  return {
    model: MODEL,
    params: { ...DEFAULT_PARAMS },
    tools: [],
    at: 0,
    shape: { from: 'm0', summary: null, tools: TOOLS },
    ...extra,
  }
}

/** user, answered (counted by the server at 5000), user, placeholder. */
function conv(answered: Partial<Message> = {}, overrides: Partial<Conversation> = {}): Conversation {
  const messages = [
    msg(0),
    msg(1, { model: MODEL, run: run(), usage: { promptTokens: 5000, completionTokens: 150 }, ...answered }),
    msg(2),
    msg(3, { content: [{ type: 'text', text: '' }], streaming: true }),
  ]
  return {
    id: 'c1',
    title: 'anchor',
    messages,
    leafId: 'm3',
    model: MODEL,
    systemPrompt: '',
    params: { ...DEFAULT_PARAMS },
    createdAt: 0,
    updatedAt: 0,
    ...overrides,
  }
}

const PLAN = { from: 0 }
function now(c: Conversation, tools = TOOLS) {
  return { model: MODEL, shape: promptShape(c, PLAN, tools), pending: 'm3' }
}

describe('replyPrompt: when the last server count is trusted', () => {
  it('takes the counted prompt as exact and estimates only what is newer', () => {
    const c = conv()
    // the answer re-sent (204) + the new question (204) + the empty placeholder (4)
    expect(replyPrompt(c, PLAN, now(c))).toEqual({ exact: 5000, estimated: MSG_TOKENS * 2 + 4 })
  })

  it('charges the counted output when it is larger than the text estimate', () => {
    const c = conv({ usage: { promptTokens: 5000, completionTokens: 900 } })
    expect(replyPrompt(c, PLAN, now(c)).estimated).toBe(904 + MSG_TOKENS + 4)
  })

  it('anchors a Continue on the turn being extended', () => {
    const c = conv()
    c.messages = c.messages.slice(0, 2)
    c.leafId = 'm1'
    expect(replyPrompt(c, PLAN, { ...now(c), pending: 'm1' })).toEqual({
      exact: 5000,
      estimated: MSG_TOKENS,
    })
  })
})

describe('replyPrompt: when it falls back to the full estimate', () => {
  const full = (c: Conversation) => ({ exact: 0, estimated: promptTokensFrom(c, 0) })

  it('has no counted turn to lean on', () => {
    const c = conv({ usage: undefined })
    expect(replyPrompt(c, PLAN, now(c))).toEqual(full(c))
  })

  it('was counted by another model, whose tokenizer is not this one', () => {
    const c = conv({ run: run({ model: 'cloud/other' }) })
    expect(replyPrompt(c, PLAN, now(c))).toEqual(full(c))
  })

  it('carried different tools, so the schemas in the count are not the ones riding now', () => {
    const c = conv()
    expect(replyPrompt(c, PLAN, now(c, 'github#c'))).toEqual(full(c))
  })

  it('ran under a different system prompt', () => {
    const c = conv({ run: run({ systemPrompt: 'be brief' }) })
    expect(replyPrompt(c, PLAN, now(c))).toEqual(full(c))
  })

  it('was sent from a different first message, or under another summary', () => {
    const moved = conv({ run: run({ shape: { from: 'm9', summary: null, tools: TOOLS } }) })
    expect(replyPrompt(moved, PLAN, now(moved))).toEqual(full(moved))
    const summarized = conv({ run: run({ shape: { from: 'm0', summary: 'm0', tools: TOOLS } }) })
    expect(replyPrompt(summarized, PLAN, now(summarized))).toEqual(full(summarized))
  })

  it('called tools: its usage is the sum of every round, a cost and not a size', () => {
    const called = conv({
      toolCalls: [{ id: 't', serverLabel: 'github', name: 'x', arguments: '{}', status: 'completed' }],
    })
    expect(replyPrompt(called, PLAN, now(called))).toEqual(full(called))
    const searched = conv({
      webSearches: [{ id: 'w', query: 'q', status: 'completed', sources: [] }],
    })
    expect(replyPrompt(searched, PLAN, now(searched))).toEqual(full(searched))
  })

  it('is a compare lane, whose history is filtered per model', () => {
    const c = conv({ group: 'g1' })
    expect(replyPrompt(c, PLAN, now(c))).toEqual(full(c))
  })

  it('was recorded before shapes existed, or by the macOS app', () => {
    const c = conv({ run: run({ shape: undefined }) })
    expect(replyPrompt(c, PLAN, now(c))).toEqual(full(c))
  })
})

describe('windowRemaining: the margin follows what is uncertain', () => {
  it('keeps the flat slack over a prompt that is all estimate', () => {
    expect(windowRemaining(16_000, 5412)).toBe(16_000 - 5412 - 1024)
  })

  it('charges a quarter of the estimated remainder, never under 128, over an exact count', () => {
    // 412 estimated -> margin max(128, 103) = 128
    expect(windowRemaining(16_000, 412, undefined, 5000)).toBe(16_000 - 5412 - 128)
    // 4000 estimated -> margin 1000
    expect(windowRemaining(32_000, 4000, undefined, 5000)).toBe(32_000 - 9000 - 1000)
  })

  it('never asks for more than is really left, and refuses under a useful reply', () => {
    expect(windowRemaining(8192, 100, undefined, 7700)).toBe(392)
    expect(windowRemaining(8192, 100, undefined, 8092 - MIN_USEFUL_REPLY)).toBe(MIN_USEFUL_REPLY)
    expect(windowRemaining(8192, 100, undefined, 8093 - MIN_USEFUL_REPLY)).toBe(0)
    expect(windowRemaining(8192, 9000)).toBe(0)
  })

  it('reads only a finite positive ceiling as a ceiling', () => {
    for (const none of [undefined, 0, -5, NaN, Infinity]) {
      expect(windowRemaining(16_000, 1000, none)).toBe(16_000 - 1000 - 1024)
    }
    expect(windowRemaining(16_000, 1000, 2048)).toBe(2048)
    expect(windowRemaining(0, 1000, 1500.9)).toBe(1500)
  })
})

describe('parseContextOverflow: a refusal that states its numbers', () => {
  it('reads Anthropic: input + max_tokens > limit', () => {
    expect(
      parseContextOverflow(
        'input length and `max_tokens` exceed context limit: 190000 + 21333 > 200000, decrease input length or `max_tokens` and try again',
      ),
    ).toEqual({ input: 190_000, limit: 200_000 })
  })

  it('reads the OpenAI-style message vLLM also writes', () => {
    expect(
      parseContextOverflow(
        "This model's maximum context length is 8192 tokens. However, you requested 8500 tokens (7988 in the messages, 512 in the completion). Please reduce the length of the messages or completion.",
      ),
    ).toEqual({ limit: 8192, input: 7988 })
  })

  it('leaves anything else alone', () => {
    expect(parseContextOverflow('Invalid API key')).toBeNull()
    expect(parseContextOverflow('context length exceeded')).toBeNull()
    expect(parseContextOverflow('')).toBeNull()
  })
})
