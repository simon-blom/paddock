// The image-generation wire client - `/v1/images/generations` in its GPT-image
// form, through the manager relay. Two shapes: one JSON body with every
// picture when the render is done, or (`stream`) the SSE stream that carries
// progressive previews as `image_generation.partial_image` events and the
// final picture as `image_generation.completed`. The send path in
// useChatStream asks for the stream whenever the endpoint offers previews
// and one picture is wanted; several pictures per turn go the JSON way,
// which is the API's own rule.

import type { ImageParams } from '@/types/chat'

export interface ImageWire {
  b64: string
  /** what the server produced - `opaque` | `transparent` */
  background: string
  format: string
  size: string
  quality: string
}

export interface ImageGenResult {
  images: ImageWire[]
  usage?: { inputTokens: number; outputTokens: number }
  /** wall-clock ms, send to done */
  ms: number
  /** ms to the first preview, when one arrived */
  firstPreviewMs?: number
  previews: number
}

export interface ImageGenOpts {
  /** `/v1/images/generations` through the relay - or, with `references`,
   *  `/v1/images/edits`, which takes the same fields as a multipart form
   *  plus the pictures */
  url: string
  model: string
  prompt: string
  params: ImageParams
  /** the seed actually sent - the caller resolves `random` before the call so
   *  the record and the request agree */
  seed: number
  /** progressive previews (0 = the JSON path) */
  previews: number
  /** reference pictures for an EDIT, in order; the endpoint sizes each to
   *  its own shape at the output area and takes the last one's shape for
   *  the output when `size` is auto */
  references?: Blob[]
  signal?: AbortSignal
  /** a preview landed: its index and the picture as a data URL, ready for an
   *  `<img>` - the caller shows it in the turn while the render goes on */
  onPreview?: (index: number, dataUrl: string) => void
}

/** The request body, from the chat's settings. `auto` fields are omitted so
 *  the endpoint's own defaults apply rather than a guess made here - the same
 *  untouched-means-absent rule the sampler keeps. */
export function imageRequestBody(o: ImageGenOpts): Record<string, unknown> {
  const p = o.params
  const body: Record<string, unknown> = {
    model: o.model,
    prompt: o.prompt,
    n: o.previews > 0 ? 1 : p.n,
    seed: o.seed,
    output_format: p.format,
  }
  if (p.size !== 'auto') body.size = p.size
  if (p.quality !== 'auto') body.quality = p.quality
  if (p.steps) body.steps = p.steps
  if (p.background !== 'auto') body.background = p.background
  if (o.previews > 0) {
    body.stream = true
    body.partial_images = o.previews
  }
  return body
}

/** A fresh seed in the range the API takes. `Math.random` is fine here: the
 *  seed's job is to be recorded, not to be unpredictable. */
export function randomSeed(): number {
  return Math.floor(Math.random() * 2 ** 31)
}

/** Automatic text-only variations retain their thread's seed. Reference
 * edits retain the picture through conditioning, not through its old noise:
 * reusing that noise can corrupt edits of a generated picture. Pinned seeds
 * remain exact; compare lanes share their draw through the group id. */
export function resolveImageSeed(
  policy: ImageParams['seed'], previous: number | undefined,
  group: string | undefined, editing: boolean, draw = randomSeed,
): number {
  if (typeof policy === 'number') return policy
  if (policy === 'thread' && !editing && previous !== undefined) return previous
  if (group) {
    let h = 0x811c9dc5
    for (let i = 0; i < group.length; i++) h = Math.imul(h ^ group.charCodeAt(i), 0x01000193) >>> 0
    return h & 0x7fffffff
  }
  return draw()
}

async function refusal(res: Response): Promise<Error> {
  // The runner answers a refusal as JSON with a real sentence; surface it
  // rather than a status code (no silent failures).
  let why = `image generation failed (${res.status})`
  try {
    const j = (await res.json()) as { error?: { message?: string } }
    if (j?.error?.message) why = j.error.message
  } catch {
    /* non-JSON body - keep the status line */
  }
  return new Error(why)
}

interface StreamEvent {
  type?: string
  b64_json?: string
  background?: string
  output_format?: string
  quality?: string
  size?: string
  partial_image_index?: number
  usage?: { input_tokens?: number; output_tokens?: number }
  error?: { message?: string }
}

export async function generateImages(o: ImageGenOpts): Promise<ImageGenResult> {
  const started = performance.now()
  const body = imageRequestBody(o)
  const refs = o.references ?? []
  // an edit is the same request as a form: every field a string, the
  // pictures as `image[]` parts (the SDKs' spelling)
  const res = refs.length
    ? await fetch(o.url, {
        method: 'POST',
        body: (() => {
          const form = new FormData()
          for (const [k, v] of Object.entries(body)) form.append(k, String(v))
          refs.forEach((blob, i) => form.append('image[]', blob, `reference-${i + 1}.png`))
          return form
        })(),
        signal: o.signal,
      })
    : await fetch(o.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: o.signal,
      })
  if (!res.ok) throw await refusal(res)

  if (!body.stream) {
    const j = (await res.json()) as {
      data?: { b64_json?: string }[]
      background?: string
      output_format?: string
      quality?: string
      size?: string
      usage?: { input_tokens?: number; output_tokens?: number }
    }
    const images: ImageWire[] = (j.data ?? [])
      .filter((d) => d.b64_json)
      .map((d) => ({
        b64: d.b64_json as string,
        background: j.background ?? 'opaque',
        format: j.output_format ?? o.params.format,
        size: j.size ?? '',
        quality: j.quality ?? '',
      }))
    return {
      images,
      usage: j.usage
        ? { inputTokens: j.usage.input_tokens ?? 0, outputTokens: j.usage.output_tokens ?? 0 }
        : undefined,
      ms: performance.now() - started,
      previews: 0,
    }
  }

  if (!res.body) throw new Error('the image stream ended before it began')
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  let done: StreamEvent | undefined
  let previews = 0
  let firstPreviewMs: number | undefined
  for (;;) {
    const { value, done: fin } = await reader.read()
    if (fin) break
    buf += dec.decode(value, { stream: true })
    // SSE frames are blank-line separated; the tail of `buf` is usually a
    // partial frame. An `event: error` frame is the runner's refusal mid-way.
    let cut: number
    while ((cut = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, cut)
      buf = buf.slice(cut + 2)
      let name = ''
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) name = line.slice(6).trim()
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (!payload || payload === '[DONE]') continue
        let ev: StreamEvent
        try {
          ev = JSON.parse(payload) as StreamEvent
        } catch {
          continue
        }
        if (name === 'error' || ev.error) {
          throw new Error(ev.error?.message ?? 'the render failed')
        }
        if (ev.type === 'image_generation.partial_image' && ev.b64_json) {
          previews++
          if (firstPreviewMs === undefined) firstPreviewMs = performance.now() - started
          o.onPreview?.(
            ev.partial_image_index ?? previews - 1,
            `data:image/${ev.output_format ?? o.params.format};base64,${ev.b64_json}`,
          )
        } else if (ev.type === 'image_generation.completed') {
          done = ev
        }
      }
    }
  }
  if (!done?.b64_json) throw new Error('the image stream ended without a picture')
  return {
    images: [
      {
        b64: done.b64_json,
        background: done.background ?? 'opaque',
        format: done.output_format ?? o.params.format,
        size: done.size ?? '',
        quality: done.quality ?? '',
      },
    ],
    usage: done.usage
      ? { inputTokens: done.usage.input_tokens ?? 0, outputTokens: done.usage.output_tokens ?? 0 }
      : undefined,
    ms: performance.now() - started,
    firstPreviewMs,
    previews,
  }
}

/** Base64 -> Blob, for the attachments table. */
export function blobOf(b64: string, mime: string): Blob {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return new Blob([bytes], { type: mime })
}

/** A small JPEG thumbnail of a picture, as a data URL, plus its dimensions -
 *  what the conversation document keeps inline so the bubble draws at once.
 *  The full bytes live in the attachments table; a 1024^2 PNG inlined into
 *  every debounced save would bloat the document by a megabyte a picture. */
export async function thumbnailOf(
  blob: Blob,
  edge = 320,
): Promise<{ thumb: string; width: number; height: number }> {
  const bmp = await createImageBitmap(blob)
  const m = Math.max(bmp.width, bmp.height)
  const s = m > edge ? edge / m : 1
  const c = document.createElement('canvas')
  c.width = Math.max(1, Math.round(bmp.width * s))
  c.height = Math.max(1, Math.round(bmp.height * s))
  c.getContext('2d')?.drawImage(bmp, 0, 0, c.width, c.height)
  const out = { thumb: c.toDataURL('image/jpeg', 0.8), width: bmp.width, height: bmp.height }
  bmp.close()
  return out
}
