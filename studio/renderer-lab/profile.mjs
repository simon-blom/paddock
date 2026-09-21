/** Isolated app runner + external process sampler. Never targets Paddock.app. */
import { spawn, execFileSync } from 'node:child_process'
import { mkdir, readFile, writeFile, open, stat } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { resolve } from 'node:path'
import { MemoryTrace } from './memory-trace.mjs'
import { finalMemoryIdleSeconds, reopenCheckpoint, screenLockState } from './profile-guards.mjs'

const root = fileURLToPath(new URL('../../', import.meta.url))
const executable = resolve(root, 'apps/macos/.build/PaddockRenderingLab.app/Contents/MacOS/PaddockRendererLab')
const sampler = resolve(root, 'apps/macos/.build/renderer-sampler')
const plan = process.argv[2] ?? 'scalepdf'
const idleArg = process.argv.find(arg => arg.startsWith('--idle-seconds='))
const idleSeconds = idleArg ? Number(idleArg.split('=')[1]) : 10
if (!Number.isInteger(idleSeconds) || idleSeconds < 10 || idleSeconds > 120) throw new Error('Idle observation must be 10-120 seconds')
if (!['scalepdf', 'scalegraph', 'scaledocx', 'scalerepeat', 'scalegraphlimit', 'scalegraphrepeat', 'scalegraphhover', 'scalepdfwalk', 'scalepdfrepeat', 'scalestream', 'scalestreamrepeat', 'scalestreamlong', 'all'].includes(plan)) throw new Error('Unknown profile plan')
const output = resolve(process.argv[3] ?? `/tmp/paddock-renderer-${plan}-${Date.now()}`)
const screenAtStart = screenLockState()
if (screenAtStart.locked) throw new Error('Unlock the Mac before a visual profile; WebKit suspends rendering frames while locked')
// Refuse overwrite: partial results are evidence too.
try { await stat(output); throw new Error(`Output already exists: ${output}`) } catch (e) { if (e.code !== 'ENOENT') throw e }
await mkdir(output, { recursive: true })
const processList = () => execFileSync('/bin/ps', ['-axo', 'pid=,comm='], { encoding: 'utf8' }).trim().split('\n').map(line => {
  const match = line.trim().match(/^(\d+)\s+(.+)$/); return match ? { pid: Number(match[1]), path: match[2] } : null
}).filter(Boolean)
const before = processList()
const reopenTracing = process.argv.includes('--trace-reopens')
if (reopenTracing && plan !== 'scalegraphrepeat') throw new Error('--trace-reopens requires scalegraphrepeat')
const finalMemoryTracing = process.argv.includes('--trace-final-memory')
const finalMemoryIdle = finalMemoryTracing ? finalMemoryIdleSeconds(plan, idleSeconds) : 0
const memoryTrace = process.argv.includes('--trace-memory') || reopenTracing || finalMemoryTracing ? new MemoryTrace(output, before, processList) : null
const heapSnapshot = process.argv.includes('--heap-snapshot')
const layerTracing = process.argv.includes('--trace-layers')
if (reopenTracing && (heapSnapshot || layerTracing || process.argv.includes('--trace-memory'))) throw new Error('Reopen inspection must run separately from other native/Inspector capture')
if (finalMemoryTracing && (reopenTracing || heapSnapshot || layerTracing || process.argv.includes('--trace-memory') || process.argv.includes('--sample-stacks'))) throw new Error('Final-cutoff capture must be the only inspection mode')
if (layerTracing && plan !== 'scalegraph') throw new Error('--trace-layers requires the scalegraph plan')
if (layerTracing && (memoryTrace || heapSnapshot)) throw new Error('Layer inspection must run separately from native/heap capture')
if (heapSnapshot && !memoryTrace) throw new Error('--heap-snapshot requires --trace-memory; GC must follow the unforced memory plateau')
if (before.some(p => p.path === executable)) throw new Error('Quit the existing standalone Rendering Lab first; product Paddock stays running')
execFileSync('/usr/bin/xcrun', ['swiftc', '-O', resolve(root, 'apps/macos/scripts/renderer-sampler.swift'), '-o', sampler], { stdio: 'inherit' })
const powerAtStart = execFileSync('/usr/bin/pmset', ['-g', 'batt'], { encoding: 'utf8' }).trim()
const log = await open(`${output}/app.log`, 'wx')
const samples = await open(`${output}/memory.jsonl`, 'wx')
const child = spawn(executable, ['--run', plan, '--no-snapshots', '--external-guard', '--output', output, ...(heapSnapshot ? ['--heap-snapshot'] : []), ...(layerTracing ? ['--trace-layers'] : [])], { stdio: ['ignore', log.fd, log.fd] })
let appExit = null; child.on('exit', (code, signal) => { appExit = { code, signal } })
child.on('error', error => { appExit = { error: error.message } })
const probe = spawn(sampler, [String(child.pid), before.map(p => p.pid).join(',')], { stdio: ['ignore', samples.fd, log.fd] })
let probeExit = null; probe.on('exit', code => { probeExit = code })
probe.on('error', () => { probeExit = -1 })
const started = Date.now()
let outcome = 'timeout'; let report; let lastPhase = ''
let stoppedAt
const stackSampling = process.argv.includes('--sample-stacks')
let stackStarted = false
let stackProcess
let memoryBaseline = false
let finalMemoryStarted
const tracedReopens = new Set()
let heapCapture
let layerCapture
// Keep the existing external guard effective during longer idle observations.
async function idleUntil(deadline) {
  while (Date.now() < deadline) {
    if (appExit) throw new Error('Lab exited during idle observation')
    if (probeExit !== null) throw new Error(`Sampler stopped during idle observation (${probeExit})`)
    await new Promise(r => setTimeout(r, Math.min(500, deadline - Date.now())))
  }
}
try {
  while (Date.now() - started < 900000) {
    if (appExit) { outcome = 'app-exited-before-completion'; break }
    if (probeExit !== null) { outcome = probeExit === 2 ? '8-GiB-process-guard' : 'sampler-exited'; break }
    try { report = JSON.parse(await readFile(`${output}/report.json`, 'utf8')) } catch { /* report not ready */ }
    if (layerTracing) {
      try { layerCapture = JSON.parse(await readFile(`${output}/layer-capture.json`, 'utf8')) } catch { /* capture not ready */ }
      if (layerCapture?.status === 'failed') { outcome = 'layer-capture-failed'; break }
    }
    const phase = report?.profile?.phase
    if (phase && phase !== lastPhase) { console.log(`${new Date().toISOString()} ${phase}`); lastPhase = phase }
    if (memoryTrace && !finalMemoryTracing && !memoryBaseline && phase === 'baseline/settled') {
      memoryBaseline = true
      await memoryTrace.capture('baseline', false, reopenTracing)
    }
    // Inspect near the original cycle cutoff without lengthening its idle.
    // footprint can suspend the process: this entire run is diagnostic only.
    const checkpoint = reopenTracing && reopenCheckpoint(report?.profile, tracedReopens)
    if (checkpoint) {
      tracedReopens.add(checkpoint)
      await memoryTrace.capture(checkpoint.replace('/', '-'), false, true)
    }
    if (phase?.endsWith('/full-canvas-layout') && Date.now() - report.profile.events.at(-1).at > 60000) {
      outcome = 'layout-deadline-60s'; break
    }
    if (stackSampling && !stackStarted && ['graph-50000/full-canvas-layout', 'graph-10000/full-canvas-layout'].includes(phase)) {
      const event = report.profile.events.at(-1)
      if (Date.now() - event.at >= (phase.startsWith('graph-50000') ? 6500 : 100)) {
        const prior = new Set(before.map(p => p.pid))
        const candidates = processList().filter(p => !prior.has(p.pid) && p.path.endsWith('/com.apple.WebKit.WebContent'))
        if (candidates.length === 1) {
          stackStarted = true
          stackProcess = spawn('/usr/bin/sample', [String(candidates[0].pid), '3', '5', '-file', `${output}/webcontent.sample.txt`], { stdio: ['ignore', log.fd, log.fd] })
          stackProcess.on('error', () => { stackStarted = false })
        }
      }
    }
    if (report && !report.running && (report.checks.length || phase === 'complete' || phase === 'timed-out')) {
      outcome = report.checks.some(c => c.status === 'fail') ? 'failed-check' : 'completed'
      // Final retained-memory plateau, with no synthetic allocation/forced GC.
      if (finalMemoryTracing) {
        // The repeat fixture includes its original cycle cutoffs. Fresh graph
        // runs still need their original external ten-second observation.
        // No inspection before this point: capture the high-memory state that
        // an intrusive per-cycle census might otherwise prevent reproducing.
        if (finalMemoryIdle) await idleUntil(Date.now() + finalMemoryIdle * 1000)
        finalMemoryStarted = Date.now()
        await memoryTrace.capture('final-cutoff', true)
      } else if (memoryTrace && !reopenTracing) {
        const idleStart = Date.now()
        for (const seconds of [0, 10, 20, 45]) {
          await idleUntil(idleStart + seconds * 1000)
          console.log(`${new Date().toISOString()} memory-trace post-${seconds}s`)
          await memoryTrace.capture(`post-${seconds}s`, seconds === 45)
        }
        if (heapSnapshot) {
          while (Date.now() - idleStart < 100000 && !appExit) {
            try { heapCapture = JSON.parse(await readFile(`${output}/heap-capture.json`, 'utf8')); break } catch { /* capture begins at 60s */ }
            await idleUntil(Date.now() + 500)
          }
          if (outcome === 'completed' && heapCapture?.status !== 'completed') outcome = 'heap-capture-failed'
        }
      } else await idleUntil(Date.now() + idleSeconds * 1000)
      if (layerTracing) {
        try { layerCapture = JSON.parse(await readFile(`${output}/layer-capture.json`, 'utf8')) } catch { /* missing is a failed capture */ }
        if (layerCapture?.status !== 'completed') outcome = 'layer-capture-failed'
      }
      break
    }
    await new Promise(r => setTimeout(r, 500))
  }
} catch (error) {
  outcome = probeExit === 2 ? '8-GiB-process-guard' : 'diagnostic-error'
  console.error(error)
} finally {
  stoppedAt = Date.now()
  // This child is the one launched above, not a process-name wildcard.
  if (!appExit) child.kill('SIGTERM')
  await new Promise(r => setTimeout(r, 6500))
  if (probeExit === null) probe.kill('SIGTERM')
  if (stackProcess?.exitCode === null) stackProcess.kill('SIGTERM')
  await log.close(); await samples.close()
  await memoryTrace?.finish()
  if (outcome === 'completed' && reopenTracing && tracedReopens.size !== 5) outcome = 'reopen-checkpoint-missed'
  if (outcome === 'completed' && memoryTrace?.rows.some(row => row.error || (row.code != null && row.code !== 0))) outcome = 'memory-trace-failed'
  const remaining = new Set(processList().map(p => p.pid))
  const raw = (await readFile(`${output}/memory.jsonl`, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse)
  const candidates = new Map()
  for (const sample of raw) for (const p of sample.processes) candidates.set(p.pid, { pid: p.pid, name: p.name, role: p.role, startAbstime: p.startAbstime, exitedWithLab: !remaining.has(p.pid) })
  const screenAtEnd = screenLockState()
  // Completion of JS checks does not qualify timings from a locked session.
  // This is conservative (lock could occur during shutdown), not continuous
  // lock monitoring. Preserve the evidence, but stop chained benchmark runs.
  if (outcome === 'completed' && screenAtEnd.locked) outcome = 'screen-lock-observed'
  await writeFile(`${output}/run.json`, JSON.stringify({ version: 1, plan, outcome, started, stoppedAt, ended: Date.now(), appPID: child.pid, automaticSnapshots: false, samplingIntervalMs: 500, stackSampling, stackStarted, memoryTracing: !!memoryTrace, reopenTracing, finalMemoryTracing, finalMemoryStarted, heapSnapshot, heapCapture, layerTracing, idleSeconds: finalMemoryTracing ? finalMemoryIdle : memoryTrace && !reopenTracing ? 45 : idleSeconds,
    powerAtStart, powerAtEnd: execFileSync('/usr/bin/pmset', ['-g', 'batt'], { encoding: 'utf8' }).trim(), screenAtStart, screenAtEnd,
    guardPerProcessBytes: 8 * 1024 ** 3, layoutDeadlineMs: 60000,
    attribution: 'New WebKit PIDs absent before controlled lab launch; verify disappearance after lab exit. Not authoritative OS coalition attribution. Shared/pre-existing processes excluded.',
    processes: [...candidates.values()], appExit, probeExit }, null, 2))
  console.log(JSON.stringify({ output, outcome, samples: raw.length, processes: [...candidates.values()] }, null, 2))
}
process.exitCode = outcome === 'completed' ? 0 : 1
