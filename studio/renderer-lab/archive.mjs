/** Preserve generated measurement evidence, refusing to overwrite an old archive. */
import { readFile, writeFile } from 'node:fs/promises'
import { gzipSync } from 'node:zlib'
import { summarize } from './summarize.mjs'
import { parseFootprint } from './memory-trace.mjs'
import { summarizeHeap } from './heap-summary.mjs'

async function optionalJSON(path) {
  try { return JSON.parse(await readFile(path, 'utf8')) } catch (e) { if (e.code !== 'ENOENT') throw e }
}

const [output, ...directories] = process.argv.slice(2)
if (!output?.endsWith('.json') || !directories.length) throw new Error('Usage: archive.mjs NEW-OUTPUT.json RUN-DIRECTORY...')
const runs = []
for (const directory of directories) {
  const report = JSON.parse(await readFile(`${directory}/report.json`, 'utf8'))
  const run = JSON.parse(await readFile(`${directory}/run.json`, 'utf8'))
  const samples = (await readFile(`${directory}/memory.jsonl`, 'utf8')).trim().split('\n').map(JSON.parse)
  let stackSample
  try { stackSample = await readFile(`${directory}/webcontent.sample.txt`, 'utf8') } catch (e) { if (e.code !== 'ENOENT') throw e }
  const memoryTrace = await optionalJSON(`${directory}/memory-trace.json`)
  const layerCapture = await optionalJSON(`${directory}/layer-capture.json`)
  const nativeMemoryFiles = {}
  for (const capture of memoryTrace?.captures ?? []) {
    if (!capture.file) continue
    if (!/^[a-z0-9-]+\.txt$/.test(capture.file)) throw new Error('Unsafe diagnostic archive filename')
    const text = await readFile(`${directory}/${capture.file}`, 'utf8')
    nativeMemoryFiles[capture.file] = text
    if (capture.kind === 'footprint' && capture.code === 0 && !capture.summary) capture.summary = parseFootprint(text)
  }
  const heapCapture = await optionalJSON(`${directory}/heap-capture.json`)
  // The first diagnostic pilot predates this field in run.json. Enrich the
  // archive from its original capture manifest without rewriting raw evidence.
  if (heapCapture) run.heapCapture = heapCapture
  const heapSnapshot = await optionalJSON(`${directory}/post-idle.heapsnapshot.json`)
  const heapSummary = heapSnapshot ? summarizeHeap(heapSnapshot) : undefined
  const secondHeapSnapshot = await optionalJSON(`${directory}/post-idle-2.heapsnapshot.json`)
  const secondHeapSummary = secondHeapSnapshot ? summarizeHeap(secondHeapSnapshot) : undefined
  runs.push({ directory, report, run, samples, stackSample, memoryTrace, layerCapture, nativeMemoryFiles, heapCapture, heapSnapshot, heapSummary, secondHeapSnapshot, secondHeapSummary,
    cpuCounterUnits: samples.every(s => Number.isFinite(s.nanosPerMachTick)) ? 'nanoseconds' : 'early sampler: Mach ticks mislabeled as ns; do not use as nanoseconds',
  })
}
const metadata = { version: 1, archivedAt: new Date().toISOString() }
await writeFile(output, JSON.stringify({ ...metadata, runs: runs.map(r => ({ directory: r.directory, ...summarize(r.report, r.run, r.samples), memoryTrace: r.memoryTrace, layerCapture: r.layerCapture, heapCapture: r.heapCapture, heapSummary: r.heapSummary, secondHeapSummary: r.secondHeapSummary })) }, null, 2), { flag: 'wx' })
await writeFile(`${output}.gz`, gzipSync(JSON.stringify({ ...metadata, runs })), { flag: 'wx' })
console.log(`Saved summary ${output} and compressed raw evidence ${output}.gz`)
