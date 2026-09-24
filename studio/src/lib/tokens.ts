// Client-side token accounting for the context gauge + pre-send trim guardrail.
// The server does not auto-truncate (Responses `truncation:"auto"` is rejected),
// so the studio must keep each prompt within the context window itself.
//
// Counts are ESTIMATES (~4 chars/token) - good enough for a fill gauge and for
// a conservative sliding window. A healthy margin absorbs the estimation error;
// the exact size of the last turn comes back in usage.input_tokens.

import type { Conversation, Message } from '@/types/chat'
import { messageText } from '@/types/chat'
import { activeMessages } from '@/lib/tree'

/** The turns this conversation would actually SEND: the branch on screen, not
 *  every branch it holds. Counting `conv.messages` would price abandoned
 *  branches into the context gauge and trim the wrong turns out of the prompt.
 *
 *  Every index in this module - `trimIndex`'s return, `promptTokensFrom`'s
 *  `from`, `summaryCount` - is an index into this array, and the same is true
 *  of the send path and the thread divider that mirror it. */
function thread(conv: Conversation): Message[] {
  return activeMessages(conv)
}

const CHARS_PER_TOKEN = 4
const PER_MESSAGE_OVERHEAD = 4 // role + delimiter tokens per message
const IMAGE_TOKENS = 512 // rough per-image vision cost
const SAFETY_MARGIN = 1024 // slack so an under-estimate never overflows the server

/** Prompt-planning headroom, separate from the generation ceiling. */
export const REPLY_RESERVE = 4096

/** Bounded headroom works even when the saved cap exceeds this model's window.
 *  It never limits generation: the remaining capacity is resolved at send. */
export function replyReserve(cap: number | null, context = 0, outputCeiling?: number): number {
  // A reply cap is not a reservation for that many tokens. Keep history space
  // on small models and don't disable compaction with an oversized preference.
  const headroom = context > 0 ? Math.max(1, Math.floor(context / 4)) : REPLY_RESERVE
  return Math.min(cap != null && cap > 0 ? cap : REPLY_RESERVE, REPLY_RESERVE, headroom,
    outputCeiling != null && outputCeiling > 0 ? outputCeiling : Infinity)
}

/** Local Paddock clamps against exact prompt/vision token counts at admission.
 * Send an explicit window-sized ceiling rather than an estimated safety margin
 * (the runner's own default for an omitted cap is the window too since
 * 2026-09-22; it was a flat 1024 before, which is why this was explicit). Cloud
 * providers may reject input+output overflow and still need windowRemaining
 * below. */
export function localOutputMaximum(maxCtx: number): number {
  return maxCtx > 0 ? maxCtx : REPLY_RESERVE
}

/** The reply cap for a request whose prompt is `promptTokens` long: everything
 *  the window has left, but never more than the model will actually emit.
 *
 *  The window half is what "model maximum" means for a local model - a GGUF
 *  has no output limit of its own, the context is the ceiling - and it beats a
 *  fixed number because it is never larger than what actually fits.
 *
 *  `outCap` is the provider's own reply ceiling (cloud models publish one; see
 *  `models.outCapFor`) and it is a different number from the window, usually
 *  far smaller. Without it the window half alone is dangerous on a big-context
 *  model: a 1M-context provider was asked for 1047543 output tokens against a
 *  prompt our estimator put at 9 and the provider's tokenizer put at 2134
 *  (tool schemas, which no client-side estimate can see), so input + output
 * crossed the window and the send died on a 400. Taking
 *  the smaller of the two makes the estimator's error harmless: 384k of output
 *  on a 1M window cannot overflow whatever the prompt turns out to be.
 *
 *  `exact` is the part of the prompt a server has already counted (see
 *  `replyPrompt`); `promptTokens` is then only what is newer and still an
 *  estimate. The margin is charged against the estimate, because that is the
 *  only part that can be wrong: a flat 1024 over a prompt we know to the token
 *  throws away a quarter of a 4K window for nothing.
 *
 *  Two rules hold in both cases, and the macOS app's `NativeReplyBudget` and
 *  the fixture the two share pin them:
 *  - the floor borrows from the margin, never from the window. A short reply
 *    is worth more than a margin, so a nearly full window still asks for
 *    `REPLY_FLOOR` - but only if that many tokens are really left. The floor
 *    used to be unconditional, which on a full window asked for 512 tokens
 *    that did not exist and left the refusal to the provider's 400.
 *  - 0 is an answer and means "refuse this turn": under `MIN_USEFUL_REPLY` of
 *    real room a send buys a sentence fragment and a second refusal on
 *    Continue, so the caller owes the user an error instead of a request. */
export function windowRemaining(
  maxCtx: number,
  promptTokens: number,
  outCap?: number,
  exact = 0,
): number {
  // A ceiling counts only as a finite positive number of tokens; anything else
  // (absent, 0, negative, NaN) reads as "this provider publishes none".
  const byModel = outCap !== undefined && Number.isFinite(outCap) && outCap > 0 ? outCap : Infinity
  if (!maxCtx) return Math.floor(Math.min(REPLY_RESERVE, byModel))
  const room = maxCtx - exact - promptTokens
  if (room < MIN_USEFUL_REPLY) return 0
  const margin = exact > 0 ? anchoredMargin(promptTokens) : windowSlack(maxCtx)
  return Math.floor(Math.min(byModel, Math.max(room - margin, Math.min(REPLY_FLOOR, room))))
}

/** Below this much real room a turn is refused rather than sent. */
export const MIN_USEFUL_REPLY = 256
/** What a nearly full window still asks for, out of its margin. */
const REPLY_FLOOR = 512

/** Margin over a prompt whose bulk is an exact server count: a quarter of the
 *  estimated remainder (the ~4 chars/token guess is off by that much on code
 *  and non-English text), and never under 128 for what no estimate sees - the
 *  new turn's template tokens, a date line that moved. */
function anchoredMargin(estimated: number): number {
  return Math.max(128, Math.ceil(estimated * 0.25))
}

/** Slack between "everything the window has left" and what we actually ask
 *  for. The flat [`SAFETY_MARGIN`] is the floor; past ~50K of window it grows
 *  with the window instead.
 *
 *  Flat 1024 was too thin at the top end and free to widen there: on a 1M
 *  window nobody is losing a reply to 20K of held-back budget, while the
 *  estimator's blind spots (tool schemas above all - 1632 real tokens the
 *  client counted as zero) do not shrink just because the window is huge. A
 *  model that publishes an `outCap` is already safe by the clamp above; this
 *  covers the ones that do not, and picks enabled before we stored it. */
function windowSlack(maxCtx: number): number {
  return Math.max(SAFETY_MARGIN, Math.round(maxCtx * 0.02))
}

/** Rough size of what a send will carry, for windowRemaining. Same estimator
 *  the trimmer uses, so the two agree.
 *
 *  `summary` is the compaction summary this send will inject - a
 *  `ContextPlan`'s own, never the stored one, because a summary held in
 *  reserve is not in the prompt and costs nothing. It is charged with its
 *  wrapper (`summaryBlock`, the one owner of that wording), once, and the
 *  messages it stands in for are already excluded by `from`.
 *
 *  Date/graph/tool content remains outside this estimate. Charging this
 *  known block reduces the omitted input; it does not provide exact tokenizer
 *  accounting or prove that a provider would otherwise reject the request. */
export function promptTokensFrom(conv: Conversation, from: number, summary?: string): number {
  let t = conv.systemPrompt ? estimateTokens(conv.systemPrompt) + PER_MESSAGE_OVERHEAD : 0
  // Keep the existing estimator's conservative per-block overhead.
  if (summary) t += estimateTokens(summaryBlock(summary)) + PER_MESSAGE_OVERHEAD
  const msgs = thread(conv)
  for (let i = Math.max(0, from); i < msgs.length; i++) t += messageTokens(msgs[i])
  return t
}

/** What a request carried besides its messages, as far as the client can name
 *  it: the first message sent, the summary standing in for the ones before it,
 *  and the tool set whose schemas rode along. Stored on the turn's run record.
 *  Two requests with the same shape share their whole prefix, which is what
 *  lets a later send trust an earlier turn's server-counted prompt. */
export interface PromptShape {
  /** id of the first message sent, null for an empty thread */
  from: string | null
  /** `summaryLastId` of the summary that rode, null when none did */
  summary: string | null
  /** the tool sources and built-in tools that rode, as one comparable string */
  tools: string
}

export function promptShape(conv: Conversation, plan: ContextPlan, tools: string): PromptShape {
  return {
    from: thread(conv)[Math.max(0, plan.from)]?.id ?? null,
    summary: plan.summary ? (conv.summaryLastId ?? null) : null,
    tools,
  }
}

/** A prompt split into what a server has counted and what is still a guess. */
export interface ReplyPrompt {
  exact: number
  estimated: number
}

/** The prompt a send will carry, anchored on the newest turn a server has
 *  already counted.
 *
 *  `usage.promptTokens` of an answered turn is the provider's own tokenizer
 *  over everything that turn's request held - the system prompt, the summary,
 *  the date line, the tool schemas no client estimate can see. When the next
 *  request has the same shape, that number is the exact size of their shared
 *  prefix, and only the answer and whatever came after it still need
 *  estimating. That is where the 1024-token slack and the unconditional floor
 *  came from: a prompt guessed end to end. With the bulk known they are not
 *  needed.
 *
 *  The anchor is refused, and the whole prompt estimated as before, unless all
 *  of this holds for the newest answered turn on screen:
 *  - same model (another model's tokenizer counts a different number);
 *  - same shape and same system prompt (otherwise the prefix is not shared);
 *  - it ran in one round. A turn that called tools reports the sum of every
 *    round's prompt - the manager's cloud loop adds them up for billing - which
 *    is a cost, not a size;
 *  - it is not a compare lane, whose history is filtered per model.
 *
 *  `pending` is the placeholder of the turn being generated, which is on the
 *  thread already and has nothing to say. On a Continue it is the turn being
 *  extended and is itself the anchor.
 *
 *  Known gap: a first turn, and any turn after the shape changed, is still a
 *  full estimate. A provider's count-tokens endpoint would make those exact as
 *  well. It is not used because it costs a round trip on every such send, not
 *  every provider has one, and the corrected resend in the send path already
 *  turns the rare miss into one extra request instead of an error. */
export function replyPrompt(
  conv: Conversation,
  plan: ContextPlan,
  now: { model: string; shape: PromptShape; pending?: string },
): ReplyPrompt {
  const msgs = thread(conv)
  for (let i = msgs.length - 1; i >= Math.max(0, plan.from); i--) {
    const a = msgs[i]
    if (a.role !== 'assistant') continue
    if (a.id === now.pending && !a.usage) continue
    const u = a.usage
    const r = a.run
    const s = r?.shape
    const usable =
      !!u &&
      u.promptTokens > 0 &&
      !!r &&
      !!s &&
      r.model === now.model &&
      (r.systemPrompt ?? '') === (conv.systemPrompt ?? '') &&
      s.from === now.shape.from &&
      s.summary === now.shape.summary &&
      s.tools === now.shape.tools &&
      !a.group &&
      !a.toolCalls?.length &&
      !a.webSearches?.length
    // only the newest answered turn is a candidate: an older one would leave
    // more to estimate than it saves
    if (!usable || !u) break
    // the answer is re-sent as input. Its text estimate or its counted output,
    // whichever is larger: over-counting costs a few tokens of reply,
    // under-counting costs a 400.
    let estimated = Math.max(messageTokens(a), u.completionTokens + PER_MESSAGE_OVERHEAD)
    for (let j = i + 1; j < msgs.length; j++) estimated += messageTokens(msgs[j])
    return { exact: u.promptTokens, estimated }
  }
  return { exact: 0, estimated: promptTokensFrom(conv, plan.from, plan.summary) }
}

/** The numbers out of a provider's "this does not fit" refusal, when it states
 *  them. Anthropic validates `input + max_tokens` against the window before it
 *  runs anything and says all three; OpenAI-style servers (vLLM among them)
 *  name the window and the prompt. Those are the provider's own tokenizer, so
 *  `limit - input` is the exact reply that fits and one corrected resend is
 *  safe. Anything unrecognised is null and the error is shown as it came. */
export function parseContextOverflow(message: string): { limit: number; input: number } | null {
  const anthropic = /exceed context limit:\s*(\d+)\s*\+\s*(\d+)\s*>\s*(\d+)/i.exec(message)
  if (anthropic) return { input: Number(anthropic[1]), limit: Number(anthropic[3]) }
  const openai =
    /maximum context length is (\d+) tokens.*?requested (\d+) tokens \((\d+) in the messages/is.exec(
      message,
    )
  if (openai) return { limit: Number(openai[1]), input: Number(openai[3]) }
  return null
}

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN)
}

function messageTokens(m: Message): number {
  let t = estimateTokens(messageText(m)) + PER_MESSAGE_OVERHEAD
  for (const p of m.content) if (p.type === 'image') t += IMAGE_TOKENS
  return t
}

/**
 * Best estimate of the whole thread's prompt size (system + every message +
 * optional composer draft). Calibrated by the last real usage when present:
 * an assistant turn's usage.promptTokens is the exact prompt the server saw,
 * so we anchor on it and only estimate anything newer.
 */
export function contextTokens(conv: Conversation | null | undefined, draft = ''): number {
  if (!conv) return draft ? estimateTokens(draft) : 0
  const msgs = thread(conv)
  const draftCost = draft ? estimateTokens(draft) + PER_MESSAGE_OVERHEAD : 0

  for (let i = msgs.length - 1; i >= 0; i--) {
    const u = msgs[i].usage
    if (msgs[i].role === 'assistant' && u?.promptTokens) {
      // real prompt + the re-sent answer (reasoning is not re-sent), plus any
      // messages newer than this turn, plus the draft.
      let t = u.promptTokens + (u.completionTokens ?? 0) + PER_MESSAGE_OVERHEAD
      for (let j = i + 1; j < msgs.length; j++) t += messageTokens(msgs[j])
      return t + draftCost
    }
  }

  let t = conv.systemPrompt ? estimateTokens(conv.systemPrompt) + PER_MESSAGE_OVERHEAD : 0
  for (const m of msgs) t += messageTokens(m)
  return t + draftCost
}

/**
 * The index of the first message to INCLUDE in the prompt so that
 * prompt + reply-cap fit the context. Messages before it are dropped (sliding
 * window keeping the most recent; the system prompt is always kept). Returns 0
 * when nothing needs trimming or limits are unknown. `reserve` sets aside
 * budget for extra prompt content (the injected summary).
 */
export function trimIndex(conv: Conversation, maxCtx: number, maxReply: number, reserve = 0): number {
  if (!maxCtx) return 0
  const budget = maxCtx - maxReply - SAFETY_MARGIN - reserve
  if (budget <= 0) return 0

  const sys = conv.systemPrompt ? estimateTokens(conv.systemPrompt) + PER_MESSAGE_OVERHEAD : 0
  const msgs = thread(conv)
  let used = sys
  let first = msgs.length ? msgs.length - 1 : 0

  for (let j = msgs.length - 1; j >= 0; j--) {
    const cost = messageTokens(msgs[j])
    // always keep the most recent message even if it alone is huge
    if (j < msgs.length - 1 && used + cost > budget) break
    used += cost
    first = j
  }
  return first
}

// ── context compaction (summarize-instead-of-drop) ──────────────────────────

/** Compact once the thread crosses this fraction of the prompt budget. */
const COMPACT_AT = 0.7
/** After compacting, keep roughly this fraction of the budget as raw recent
 *  messages; everything older folds into the summary. */
const KEEP_TAIL = 0.35

/** The stored summary still matches the thread: its boundary message is where
 *  it was when the summary was written. */
export function summaryValid(conv: Conversation): boolean {
  return !!(
    conv.summary &&
    conv.summaryCount &&
    thread(conv)[conv.summaryCount - 1]?.id === conv.summaryLastId
  )
}

/** The summary exactly as it rides in the prompt: the send path puts this
 *  string in `instructions` and `promptTokensFrom` charges this string, so
 *  one owner of the wording. Two owners is how the budget came to ignore text
 *  the request had been carrying all along. */
export function summaryBlock(summary: string): string {
  return `Summary of the earlier part of this conversation (older messages were compacted):\n${summary}`
}

/** What the next prompt should contain: raw messages from `from` on, preceded
 *  by `summary` when one applies. Falls back to the plain sliding window when
 *  summaries are off, absent, or stale. */
export interface ContextPlan {
  from: number
  summary?: string
}

export function planContext(
  conv: Conversation,
  maxCtx: number,
  maxReply: number,
  useSummary: boolean,
): ContextPlan {
  if (!useSummary || !summaryValid(conv)) {
    return { from: trimIndex(conv, maxCtx, maxReply) }
  }
  const summary = conv.summary as string
  const covered = conv.summaryCount as number
  // Nothing needs to give way yet -> send everything raw (the summary stays in
  // reserve until the window actually forces a choice).
  if (trimIndex(conv, maxCtx, maxReply) === 0) return { from: 0 }
  const reserve = estimateTokens(summary) + PER_MESSAGE_OVERHEAD
  const from = trimIndex(conv, maxCtx, maxReply, reserve)
  // The summary replaces its covered prefix. If even that doesn't fit, raw
  // messages past the coverage still drop off (and the thread divider says so).
  return { from: Math.max(from, covered), summary }
}

// ── server-side compaction (local lanes send context_management) ────────────

/** The stored compaction item still matches the thread: its tail-start
 *  message (the newest user message of the compacted request) is still
 *  present. Same anchor-by-id safety as `summaryValid`. */
export function serverCompactionValid(conv: Conversation): boolean {
  const sc = conv.serverCompaction
  return !!sc && thread(conv).some((m) => m.id === sc.tailStartId)
}

/** The `compact_threshold` a local lane arms `context_management` with:
 *  the same 70%-of-budget trigger the client-side compactor uses, but in the
 *  server's exact rendered tokens. Well under the window deliberately - the
 *  summarization pass reads the whole prompt, so compaction must fire while
 *  everything still fits. 0 = the window is too small to manage (caller
 *  falls back to the client plan). */
export function serverCompactThreshold(maxCtx: number, maxReply: number): number {
  const budget = maxCtx - maxReply - SAFETY_MARGIN
  if (budget <= 0) return 0
  return Math.max(512, Math.floor(budget * COMPACT_AT))
}

/** How many leading messages the next compaction should cover. 0 = the thread
 *  hasn't crossed the threshold (or there's nothing new to fold in). */
export function compactionTarget(conv: Conversation, maxCtx: number, maxReply: number): number {
  if (!maxCtx || thread(conv).length < 4) return 0
  const budget = maxCtx - maxReply - SAFETY_MARGIN
  if (budget <= 0) return 0
  if (contextTokens(conv) < budget * COMPACT_AT) return 0

  // Keep the newest messages that fit the tail allowance; cover the rest.
  const msgs = thread(conv)
  let used = 0
  let keepFrom = msgs.length
  for (let j = msgs.length - 1; j >= 0; j--) {
    used += messageTokens(msgs[j])
    if (used > budget * KEEP_TAIL && keepFrom < msgs.length) break
    keepFrom = j
  }
  // Always keep the latest exchange raw; only report growth over what the
  // current summary already covers.
  const target = Math.min(keepFrom, msgs.length - 2)
  const existing = summaryValid(conv) ? (conv.summaryCount as number) : 0
  return target > existing ? target : 0
}
