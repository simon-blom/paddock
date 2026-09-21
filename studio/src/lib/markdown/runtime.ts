import { WorkerRpc } from './worker-rpc'
import type { ParseInput, ParseOutput } from './parse'
import type { HighlightInput } from './syntax'

export const markdownParser = new WorkerRpc<ParseInput, ParseOutput>(
  () => new Worker(new URL('./parse.worker.ts', import.meta.url), { type: 'module', name: 'paddock-markdown' }),
  // A reused WebKit worker can sleep ~1s before starting the next job. An
  // explicit start acknowledgement distinguishes that from slow parsing:
  // recover an unstarted job once, but keep healthy workers/caches warm.
  // A 5s idle election was tested and rejected after worse reopen memory.
  { maxJobs: 32, maxBytes: 8 * 1024 * 1024, timeoutMs: 15000, idleMs: 15000, startTimeoutMs: 100 },
)
export const syntaxHighlighter = new WorkerRpc<HighlightInput, string>(
  () => new Worker(new URL('./syntax.worker.ts', import.meta.url), { type: 'module', name: 'paddock-syntax' }),
)
