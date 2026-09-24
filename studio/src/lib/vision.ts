// What an attached image costs the model, from the budget the endpoint
// publishes (`vision_budget` on /api/server and /v1/models) rather than
// anything guessed here. The engine computes it from the mmproj it loaded, so
// these numbers are the tower's own, not a table that rots when a model lands.

/** OpenAI's `detail` knob, sent verbatim on each image part. */
export type ImageDetail = 'auto' | 'low' | 'high'

export interface VisionBudget {
  max_pixels: number
  min_pixels: number
  /** longest usable edge when the family bounds one (granite does, at 3840) */
  max_edge: number | null
  pixels_per_token: number
  max_tokens: number
  min_tokens: number
  /** what `auto` resolves to on this model */
  auto_max_tokens: number
}

/** Vision rows `detail` allows on one image here. */
export function tokenCap(b: VisionBudget, d: ImageDetail): number {
  if (d === 'high') return b.max_tokens
  if (d === 'low') return b.min_tokens
  return b.auto_max_tokens
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(Math.max(n, lo), hi)
}

/** Source pixels a row count is worth, clamped to what the tower will do. */
export function pixelsForTokens(b: VisionBudget, tokens: number): number {
  const t = clamp(tokens, b.min_tokens, b.max_tokens)
  return clamp(t * b.pixels_per_token, b.min_pixels, b.max_pixels)
}

/** Fit (w, h) under a pixel ceiling, aspect preserved, edge cap first. Mirrors
 *  VisionBudget::fit_px in the engine, floors for the same reason: rounding a
 *  scaled-down pair can land back over the ceiling it just enforced. */
function fitPx(b: VisionBudget, w: number, h: number, maxPx: number): [number, number] {
  let fw = Math.max(w, 1)
  let fh = Math.max(h, 1)
  if (b.max_edge && Math.max(fw, fh) > b.max_edge) {
    const s = b.max_edge / Math.max(fw, fh)
    fw *= s
    fh *= s
  }
  const px = fw * fh
  if (px > maxPx) {
    const s = Math.sqrt(maxPx / px)
    fw *= s
    fh *= s
  }
  return [Math.max(Math.floor(fw), 1), Math.max(Math.floor(fh), 1)]
}

/** Vision rows an image of (w, h) costs at `detail` on this endpoint. */
export function tokensFor(b: VisionBudget, w: number, h: number, d: ImageDetail): number {
  const [fw, fh] = fitPx(b, w, h, pixelsForTokens(b, tokenCap(b, d)))
  const n = Math.ceil((fw * fh) / Math.max(b.pixels_per_token, 1))
  return clamp(n, b.min_tokens, b.max_tokens)
}

/** "1.2k" / "480" - a row count next to a menu label, not a precise figure. */
export function formatTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k` : String(n)
}

/** Plain-language menu labels. No jargon: a person picking a picture size
 *  should not have to know what a vision token is to choose well. */
export const DETAIL_LABEL: Record<ImageDetail, string> = {
  high: 'Original',
  auto: 'Auto-resize',
  low: 'Smaller',
}

export const DETAIL_HINT: Record<ImageDetail, string> = {
  high: 'Use full-detail input. Model limits still apply; processing can take longer.',
  auto: 'Fit large images for analysis. Your original is kept.',
  low: 'Fewest tokens, fastest. Fine when you only need the gist.',
}

export const DETAIL_ORDER: ImageDetail[] = ['auto', 'high', 'low']
