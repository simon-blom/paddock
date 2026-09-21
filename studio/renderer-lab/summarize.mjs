import { readFile, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

export function summarize(report, run, samples) {
  // Inspector attachment and heap snapshots alter GC and add their own WebKit
  // processes. Never fold those later diagnostic samples into the idle bar.
  if (run.heapCapture?.started) samples = samples.filter(s => s.at < run.heapCapture.started)
  if (run.finalMemoryStarted) samples = samples.filter(s => s.at < run.finalMemoryStarted)
  const verified = new Set(run.processes.filter(p => p.exitedWithLab).map(p => p.pid))
  const usable = sample => sample.processes.filter(p => verified.has(p.pid))
  const mib = bytes => Math.round(bytes / 1048576 * 10) / 10
  const last = rows => rows.at(-1)
  const phaseRows = report.profile.events.map((event, index, events) => {
    const end = events[index + 1]?.at ?? run.stoppedAt ?? run.ended
    const rows = samples.filter(s => s.at >= event.at && s.at < end && usable(s).length && (run.appPID == null || s.processes.some(p => p.pid === run.appPID)))
    const totals = rows.map(s => usable(s).reduce((n, p) => n + p.footprint, 0))
    return {
      phase: event.phase, samples: rows.length,
      // Concurrent sum, not sum of each process's independently reached peak.
      sampledPeakMiB: totals.length ? mib(Math.max(...totals)) : null,
      lastMiB: totals.length ? mib(last(totals)) : null,
      processes: [...new Set(rows.flatMap(s => usable(s).map(p => p.name)))].map(name => {
        const points = rows.flatMap(s => usable(s).filter(p => p.name === name))
        return { name, sampledPeakMiB: mib(Math.max(...points.map(p => p.footprint))), lastMiB: mib(last(points).footprint) }
      }),
      metrics: report.profile.metrics.filter(m => m.phase === event.phase),
    }
  })
  return { version: 1, run, environment: report.environment, checks: report.checks,
    limitations: [
      '500 ms sampling can miss transients; per-process lifetime peaks are separate and not additive.',
      'Launch-delta/exit-correlated attribution is not OS coalition accounting; pre-existing/shared processes excluded.',
      'Frame gaps and DOM/next-rAF proxies are not actual screen presentation timestamps.',
      'Canvas RGBA dimensions are nominal, not physical memory; WASM reservations/RSS are not footprint.',
      'Foreground synthetic workload, no inference, no screenshots.',
      ...(run.finalMemoryTracing
        ? ['Native inspection starts after the original accepted idle cutoff; inspection-era samples are excluded. No pre-cutoff native inspection.']
        : run.memoryTracing ? ['Native memory inspection is intrusive; this run is allocation diagnostics, not a latency bar.'] : []),
      ...(run.layerTracing ? ['Layer inspection attaches Inspector before admission; the ENTIRE run is diagnostic, not a memory or latency bar.'] : []),
      run.heapSnapshot ? 'Post-idle heap snapshot forces GC; inspector-era samples excluded from these summaries.' : 'No forced garbage collection.',
    ],
    spans: report.profile.spans ?? [],
    graphTriggers: report.profile.graphTriggers ?? [],
    graphTriggersDropped: report.profile.graphTriggersDropped ?? 0,
    processPeaks: run.processes.filter(p => samples.some(s => s.processes.some(q => q.pid === p.pid))).map(p => {
      const points = samples.flatMap(s => s.processes.filter(q => q.pid === p.pid))
      return { ...p, sampledPeakMiB: mib(Math.max(...points.map(p => p.footprint))), lifetimePeakMiB: mib(Math.max(...points.map(p => p.peakFootprint))) }
    }), phases: phaseRows }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  for (const directory of process.argv.slice(2)) {
    const report = JSON.parse(await readFile(`${directory}/report.json`, 'utf8'))
    const run = JSON.parse(await readFile(`${directory}/run.json`, 'utf8'))
    const samples = (await readFile(`${directory}/memory.jsonl`, 'utf8')).trim().split('\n').map(JSON.parse)
    const summary = summarize(report, run, samples)
    await writeFile(`${directory}/summary.json`, JSON.stringify(summary, null, 2))
    console.log(directory, run.outcome)
    console.table(summary.phases.map(p => ({ phase: p.phase, peakMiB: p.sampledPeakMiB, lastMiB: p.lastMiB,
      elapsed: p.metrics.find(m => m.name === 'elapsed')?.value,
      gapMax: p.metrics.find(m => m.name === 'frame-gap-max')?.value,
    })))
  }
}
