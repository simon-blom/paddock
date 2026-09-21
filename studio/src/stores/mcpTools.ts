// Tool inventories for the composer's picker: what each MCP server actually
// exposes (names + descriptions), fetched through the manager's one-shot
// listing probe (/api/mcp/tools) and cached for the session. A failed listing
// is a state, not an error - the picker still offers the whole server.
import { defineStore } from 'pinia'
import { reactive } from 'vue'

export interface McpToolInfo {
  name: string
  description?: string
}

export interface ToolListing {
  status: 'loading' | 'ok' | 'error'
  tools: McpToolInfo[]
  /** The server's MCP `instructions` - the runner folds these into the system
   *  prompt, so the Studio shows them read-only rather than leaving injected
   *  text invisible. */
  instructions?: string
  error?: string
}

/** Cache key: a server tool is `p:<port>:<label>` (resolved through the
 *  endpoint's config file), a connector is `c:<id>` (resolved through the
 *  library, tokens refreshed). */
export function serverKey(port: number, label: string): string {
  return `p:${port}:${label}`
}
export function connectorKey(id: string): string {
  return `c:${id}`
}
/** A manager-hosted first-party server (artifacts): `b:<name>`. */
export function builtinKey(name: string): string {
  return `b:${name}`
}

export const useMcpToolsStore = defineStore('mcpTools', () => {
  const listings = reactive(new Map<string, ToolListing>())
  const epochs = new Map<string, number>()

  async function fetchListing(key: string, body: Record<string, unknown>): Promise<void> {
    const epoch = (epochs.get(key) ?? 0) + 1
    epochs.set(key, epoch)
    listings.set(key, { status: 'loading', tools: [] })
    try {
      const res = await fetch('/api/mcp/tools', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const doc = (await res.json().catch(() => null)) as {
        ok?: boolean
        tools?: McpToolInfo[]
        instructions?: string
        error?: { message?: string } | string
      } | null
      if (epochs.get(key) !== epoch) return
      if (res.ok && doc?.ok && Array.isArray(doc.tools)) {
        listings.set(key, { status: 'ok', tools: doc.tools, instructions: doc.instructions })
      } else {
        const msg = typeof doc?.error === 'object' ? doc.error?.message : doc?.error
        listings.set(key, { status: 'error', tools: [], error: msg ?? `listing failed (${res.status})` })
      }
    } catch (e) {
      if (epochs.get(key) !== epoch) return
      listings.set(key, { status: 'error', tools: [], error: e instanceof Error ? e.message : String(e) })
    }
  }

  /** Kick off (or reuse) the listing for one server tool. Errors are retried
   *  on the next ensure - reopening the picker gives the server another go. */
  function ensureServer(port: number, label: string): void {
    const key = serverKey(port, label)
    const cur = listings.get(key)
    if (cur && cur.status !== 'error') return
    void fetchListing(key, { port, label })
  }

  function ensureConnector(id: string): void {
    const key = connectorKey(id)
    const cur = listings.get(key)
    if (cur && cur.status !== 'error') return
    void fetchListing(key, { connector_id: id })
  }

  /** A server the MANAGER hosts itself (artifacts). Same probe endpoint, but
   *  answered in-process, so it never fails on a cold start. */
  function ensureBuiltin(name: string): void {
    const key = builtinKey(name)
    const cur = listings.get(key)
    if (cur && cur.status !== 'error') return
    void fetchListing(key, { builtin: name })
  }

  function get(key: string): ToolListing | undefined {
    return listings.get(key)
  }

  /** Invalidate on configuration edits, not every time the picker opens.
   * Old in-flight probes cannot repopulate a stale credential/tool listing. */
  function invalidate(): void {
    for (const [key, epoch] of epochs) epochs.set(key, epoch + 1)
    listings.clear()
  }

  return { listings, ensureServer, ensureConnector, ensureBuiltin, get, invalidate }
})
