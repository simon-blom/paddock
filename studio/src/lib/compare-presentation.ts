import type { Message } from '@/types/chat'

/** Only comparable completed speech runs can win; never reward a shorter chat
 * answer or race a still-running/failed lane. Shared with native fixtures. */
export function fastestCompareLane(messages: Message[]): string | undefined {
  if (messages.length < 2 || messages.some(m => m.streaming || m.stopped || m.error || m.run?.contended)) return
  const raced = messages.filter(m => m.transcript)
  if (raced.length < 2 || raced.some(m => !Number.isFinite(m.usage?.ms) || !(m.usage!.ms! > 0))) return
  const best = raced.reduce((a, m) => m.usage!.ms! < a.usage!.ms! ? m : a)
  if (raced.filter(m => m !== best).every(m => m.usage!.ms! > best.usage!.ms! * 1.1)) return best.id
}
