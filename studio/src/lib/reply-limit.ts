import { windowRemaining } from './tokens'

// Preference/editor bounds, not an invented limit on a model's capabilities.
export const MAX_REPLY_LIMIT = 1_048_576
export const REPLY_LIMIT_ERROR = 'Enter a whole number from 1 to 1,048,576 tokens.'

export function parseReplyLimit(text: string): number | null {
  const raw = text.trim()
  if (!/^[0-9]+$/.test(raw)) return null
  const n = Number(raw)
  return Number.isSafeInteger(n) && n >= 1 && n <= MAX_REPLY_LIMIT ? n : null
}

/** Resolve a saved ceiling for one request, never rewrite the preference.
 * Unknown capacity means omit the field and use the endpoint's own default;
 * 0 means known insufficient room. Local admission owns exact tokenization. */
export function resolveReplyLimit(
  requested: number | null,
  cloud: boolean,
  context: number,
  prompt = 0,
  outputCeiling?: number,
  exact = 0,
): number | null {
  const ceiling = outputCeiling != null && Number.isSafeInteger(outputCeiling) && outputCeiling > 0
    ? outputCeiling : null
  let available = context > 0
    ? cloud ? windowRemaining(context, prompt, ceiling ?? undefined, exact) : context
    : ceiling
  if (available != null && ceiling != null) available = Math.min(available, ceiling)
  return requested == null ? available : available == null ? requested : Math.min(requested, available)
}
