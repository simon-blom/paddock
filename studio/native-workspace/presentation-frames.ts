/** Keep every WK message bounded without truncating long native transcripts.
 * Splitting UTF-8 bytes avoids splitting a surrogate pair in Swift strings. */
export function presentationFrames(value: { revision: number }, max = 64 * 1024 * 1024): unknown[] {
  const json = JSON.stringify(value), bytes = new TextEncoder().encode(json)
  if (bytes.length > max) throw new Error('Conversation presentation exceeds the 64 MiB safety limit')
  if (bytes.length <= 192 * 1024) return [JSON.parse(json)]
  const size = 48 * 1024, count = Math.ceil(bytes.length / size)
  return Array.from({ length: count }, (_, index) => {
    const part = bytes.subarray(index * size, (index + 1) * size)
    let binary = ''
    for (let i = 0; i < part.length; i += 4096) binary += String.fromCharCode(...part.subarray(i, i + 4096))
    return { transfer: value.revision, index, count, bytes: bytes.length, payload: btoa(binary) }
  })
}
