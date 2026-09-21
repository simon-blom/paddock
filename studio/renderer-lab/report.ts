import { reactive } from 'vue'

export type Command = 'all' | 'markdown' | 'math' | 'mermaid' | 'pdf' | 'docx' | 'traverse' | 'stress' | 'security' | 'reset' | 'theme' | 'scale' | 'scalepdf' | 'scalegraph' | 'scaledocx' | 'scalerepeat' | 'scalegraphlimit' | 'scalegraphrepeat' | 'scalegraphhover' | 'scalepdfwalk' | 'scalepdfrepeat' | 'scalestream' | 'scalestreamrepeat' | 'scalestreamlong'
export const profile = {
  phase: 'idle',
  events: [] as { phase: string; at: number }[],
  metrics: [] as { phase: string; name: string; value: number; unit: string }[],
  spans: [] as { phase: string; name: string; start: number; ms: number }[],
  graphTriggers: [] as object[],
  graphTriggersDropped: 0,
}
export function phase(name: string) {
  if (profile.events.length >= 256) throw new Error('Profile event budget exceeded')
  profile.phase = name; profile.events.push({ phase: name, at: Date.now() }); publish()
}
export function metric(name: string, value: number, unit = 'ms', measuredPhase = profile.phase) {
  if (!Number.isFinite(value) || profile.metrics.length >= 1024) throw new Error('Invalid or excessive profile metric')
  profile.metrics.push({ phase: measuredPhase, name, value: Math.round(value * 100) / 100, unit })
}
export interface Check { name: string; status: 'pass' | 'fail' | 'running' | 'manual'; detail: string; ms: number }
declare global {
  interface Window {
    __labDisposeShim?: boolean
    paddockLab: { command: (name: Command) => Promise<void>; memoryStats?: () => unknown }
    webkit?: { messageHandlers?: { labReport?: { postMessage: (report: unknown) => void } } }
  }
}
export const state = reactive({
  running: false,
  requiresReload: false,
  checks: [] as Check[],
  environment: {
    userAgent: navigator.userAgent,
    origin: location.origin,
    secureContext: String(isSecureContext),
    crossOriginIsolated: String(crossOriginIsolated),
    sharedArrayBuffer: String(typeof SharedArrayBuffer !== 'undefined'),
    disposalSymbolsShim: String(window.__labDisposeShim === true),
    hardwareConcurrency: String(navigator.hardwareConcurrency),
    devicePixelRatio: String(devicePixelRatio),
    assets: 'Bundled only; synthetic fixtures; no model services or product stores',
    renderers: 'Studio markstream/KaTeX/Mermaid/Shiki, Lector PDFium ST, Scriptor, Traverse ST + Studio GraphCanvas',
  },
})
export function publish(checkpoint?: Command) {
  const report = { version: 1, running: state.running, checks: state.checks, environment: state.environment, checkpoint, profile }
  window.webkit?.messageHandlers?.labReport?.postMessage(JSON.parse(JSON.stringify(report)))
}
export function reportError(name: string, error: unknown) {
  const detail = error instanceof Error ? error.stack ?? `${error.name}: ${error.message}` : String(error)
  // A broken renderer cannot fill the bounded bridge with duplicate runtime errors.
  if (state.checks.some(c => c.name === name && c.detail === detail)) return
  if (state.checks.length < 100) state.checks.push({ name, status: 'fail', detail: detail.slice(0, 4000), ms: 0 })
  publish()
}
export function assert(value: unknown, message: string): asserts value {
  if (!value) throw new Error(message)
}
export const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))
export async function until(test: () => unknown, detail: string, timeout = 12000) {
  const start = performance.now()
  while (!test()) {
    if (performance.now() - start > timeout) throw new Error(detail)
    await delay(50)
  }
}
export async function check(name: string, body: () => Promise<string>, timeout = 20000): Promise<boolean> {
  const row: Check = reactive({ name, status: 'running', detail: '', ms: 0 })
  const old = state.checks.findIndex(c => c.name === name)
  if (old >= 0) state.checks.splice(old, 1, row)
  else state.checks.push(row)
  publish()
  const start = performance.now()
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    row.detail = await Promise.race([body(), new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        state.requiresReload = true
        reject(new Error(`Timed out after ${timeout} ms; reload before retrying a stuck engine`))
      }, timeout)
    })])
    row.status = 'pass'
  } catch (error) {
    row.status = 'fail'
    row.detail = error instanceof Error ? `${error.name}: ${error.message}` : String(error)
  } finally {
    clearTimeout(timer)
    row.ms = Math.round((performance.now() - start) * 10) / 10
    publish()
  }
  return row.status === 'pass'
}
export function manual(name: string, detail: string) {
  if (!state.checks.some(c => c.name === name)) state.checks.push({ name, status: 'manual', detail, ms: 0 })
}
