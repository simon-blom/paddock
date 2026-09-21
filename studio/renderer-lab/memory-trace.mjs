/** Opt-in native allocation diagnostics for the isolated lab only.
 * These commands inspect (and can suspend) WebContent: never use their runs
 * as latency bars. No GC, pressure injection, process-wide environment or
 * debugger setting is changed. Every target must be new since lab launch.
 */
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { writeFile } from 'node:fs/promises'

const execute = promisify(execFile)

export function parseFootprint(text) {
  const categories = []
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*(\d+) B\s+(\d+) B\s+(\d+) B\s+(\d+)\s+(.+)$/)
    if (m && m[5] !== 'TOTAL') categories.push({ name: m[5], dirtyBytes: Number(m[1]), cleanBytes: Number(m[2]), reclaimableBytes: Number(m[3]), regions: Number(m[4]) })
  }
  const footprintBytes = Number(text.match(/Footprint: (\d+) B/)?.[1])
  if (!Number.isFinite(footprintBytes) || !categories.length) throw new Error('Unrecognized footprint output; expected byte-format category rows')
  return { footprintBytes, categories }
}

export class MemoryTrace {
  rows = []
  constructor(output, before, processList) {
    this.output = output
    this.prior = new Set(before.map(p => p.pid))
    this.processList = processList
  }
  async capture(label, heap = false, footprintOnly = false) {
    if (!/^[a-z0-9-]{1,40}$/.test(label)) throw new Error('Invalid memory checkpoint label')
    const targets = this.processList().filter(p => !this.prior.has(p.pid)
      && p.path.endsWith('/com.apple.WebKit.WebContent'))
    if (targets.length !== 1) {
      this.rows.push({ label, at: Date.now(), error: `Expected one new WebContent PID, found ${targets.length}; refused ambiguous attachment` })
      return
    }
    const pid = targets[0].pid
    const commands = [
      ['/usr/bin/footprint', ['-p', String(pid), '-f', 'bytes'], 'footprint'],
      ...(!footprintOnly ? [['/usr/bin/vmmap', ['-summary', '-w', String(pid)], 'vmmap']] : []),
      ...(heap ? [['/usr/bin/heap', ['-s', '--noContent', String(pid)], 'heap']] : []),
    ]
    for (const [command, args, kind] of commands) {
      const at = Date.now()
      let result
      try { result = { ...await execute(command, args, { timeout: 20000, maxBuffer: 4 * 1024 ** 2 }), code: 0 } }
      catch (error) { result = { stdout: error.stdout ?? '', stderr: error.stderr ?? '', code: error.code, error: error.message } }
      const file = `${label}-${kind}.txt`
      await writeFile(`${this.output}/${file}`, `${result.stdout}\n${result.stderr}`, { flag: 'wx' })
      let summary
      if (kind === 'footprint' && result.code === 0) {
        try { summary = parseFootprint(result.stdout) } catch (error) { result.error = error.message }
      }
      this.rows.push({ label, kind, pid, at, elapsedMs: Date.now() - at, file, code: result.code, error: result.error, summary })
    }
  }
  async finish() {
    await writeFile(`${this.output}/memory-trace.json`, JSON.stringify({ version: 1,
      diagnosticOnly: true, forcedGC: false, captures: this.rows }, null, 2), { flag: 'wx' })
  }
}
