import type { Message } from '@/types/chat'

/** Query chips only: no tool results or conversation text retained by the
 * graph worker. Native passes the same bounded, active-branch projection. */
export function restoredGraphRuns(history: readonly Message[]) {
  const runs: { cypher: string; model: string; response: null }[] = []
  let start = 0
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i]?.content?.some(p => p.type === 'graph')) { start = i; break }
  }
  for (let i = history.length - 1; i >= start && runs.length < 20; i--) {
    const message = history[i]!
    const calls = message.toolCalls ?? []
    for (let j = calls.length - 1; j >= 0 && runs.length < 20; j--) {
      const call = calls[j]!
      if (call.serverLabel !== 'graph' || call.name !== 'graph_query' || typeof call.arguments !== 'string') continue
      if (call.arguments.length > 65_536 || new TextEncoder().encode(call.arguments).length > 65_536) continue
      try {
        const cypher = (JSON.parse(call.arguments) as { cypher?: unknown } | null)?.cypher
        if (typeof cypher === 'string' && cypher.length) runs.push({ cypher, model: message.model ?? '', response: null })
      } catch { /* Incomplete tool arguments cannot be replayed. */ }
    }
  }
  return runs.reverse()
}
