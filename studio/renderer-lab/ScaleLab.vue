<script setup lang="ts">
import { nextTick, onBeforeUnmount, ref, shallowRef } from 'vue'
import Graph from 'graphology'
import { TraverseDb } from '@truespar/traverse-wasm'
import GraphCanvas from '@/components/chat/graph/GraphCanvas.vue'
import { buildGraph } from '@/lib/graph/session'
import MarkdownContent from '@/components/chat/MarkdownContent.vue'
import { PDFCase, DOCXCase } from './documents'
import { assert, check, delay, metric, phase, profile, publish, state, until } from './report'
import { StageTrace } from './trace'
import { ScriptorView } from '@truespar/scriptor-core'
import { ScriptorDoc } from '@truespar/scriptor-wasm'
import { markdownMounts } from '@/lib/markdown/mount-scheduler'
import { markdownParser } from '@/lib/markdown/runtime'

const kind = ref('idle')
const label = ref('Scale tests: cold opens, search/scroll, full graph layout, concurrent streams and repeated teardown.')
const pdfHost = ref<HTMLElement>()
const docxHost = ref<HTMLElement>()
const graphHost = ref<HTMLElement>()
const graphCanvas = ref<InstanceType<typeof GraphCanvas>>()
const graph = shallowRef<Graph | null>(null)
const lanes = ref<string[]>([])
const pdf = new PDFCase()
const docx = new DOCXCase()
let db: TraverseDb | null = null
let graphTrace: StageTrace | null = null
const traceGraph = (stage: string, start: number) => graphTrace?.record(stage, start)
const frame = () => new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
const communitySize = 50
const quantile = (values: number[], p: number) => [...values].sort((a, b) => a - b)[Math.max(0, Math.ceil(values.length * p) - 1)] ?? 0

async function measure(name: string, body: () => Promise<void>) {
  phase(name); label.value = name
  const gaps: number[] = []
  let last = performance.now(); let raf = 0
  const tick = (now: number) => {
    if (gaps.length < 100000) gaps.push(now - last)
    if (now - last > 50 && profile.spans.length < 512) profile.spans.push({ name: 'frame-gap', start: last, ms: now - last, phase: profile.phase })
    last = now; raf = requestAnimationFrame(tick)
  }
  raf = requestAnimationFrame(tick)
  const start = performance.now()
  try { await body(); await frame() }
  finally {
    cancelAnimationFrame(raf)
    metric('elapsed', performance.now() - start)
    metric('frame-gap-p95', quantile(gaps, .95)); metric('frame-gap-p99', quantile(gaps, .99))
    metric('frame-gap-max', gaps.length ? Math.max(...gaps) : 0)
    metric('frames', gaps.length, 'count'); metric('gaps-over-50ms', gaps.filter(n => n > 50).length, 'count')
    publish()
  }
}
function canvases(host: HTMLElement) {
  const all = [...host.querySelectorAll('canvas')]
  metric('canvas-count', all.length, 'count')
  // Dimensions are not allocated memory: browsers may lazily back blank canvases.
  metric('nominal-rgba-canvas-bytes', all.reduce((sum, c) => sum + c.width * c.height * 4, 0), 'bytes')
  metric('dom-elements', host.querySelectorAll('*').length, 'count')
}
async function cleanup(name: string, settle = 3000) {
  await measure(`${name}/close`, async () => {
    graph.value = null; lanes.value = []; db?.close(); db = null
    await pdf.close(); docx.close(); await nextTick()
    kind.value = 'idle'
  })
  await measure(`${name}/settled`, async () => { await delay(settle); metric('remaining-canvases', document.querySelectorAll('.scale-lab canvas').length, 'count') })
}
async function pdfCase(pages: number, images = false, name = `pdf-${images ? 'images-' : ''}${pages}`, jumps = 30) {
  kind.value = 'pdf'; await nextTick()
  try {
    await measure(`${name}/open`, async () => {
      await pdf.open(pdfHost.value!, `${images ? 'pdf-images' : 'pdf'}-${pages}.pdf`, pages, false)
      canvases(pdfHost.value!)
    })
    await measure(`${name}/loaded`, async () => { await delay(2000) })
    await measure(`${name}/search`, async () => { await pdf.search(pages); metric('matches', pages, 'count') })
    await measure(`${name}/scroll`, async () => {
      const times: number[] = []
      for (let i = 0; i < jumps; i++) {
        const page = Math.round(i * (pages - 1) / (jumps - 1)); const start = performance.now()
        pdf.pane!.viewport.scrollToPage(page, false)
        await until(() => pdf.pane!.isPageReady(page), `Page ${page + 1} never painted at current resolution`, 30000)
        await frame(); times.push(performance.now() - start); await delay(40)
        if (jumps > 30 && (i + 1) % 25 === 0) {
          metric('visited-targets-progress', i + 1, 'count')
          metric('painted-page-count-progress', pdfHost.value!.querySelectorAll('.lector-page:not(.lector-page--loading)').length, 'count')
          metric('raster-resident-bytes-progress', pdf.pane!.renderStats.residentBytes, 'bytes')
          metric('raster-shared-budget-bytes-progress', pdf.pane!.renderStats.sharedBudget.bytes, 'bytes')
          publish()
        }
      }
      metric('page-jump-p50', quantile(times, .5)); metric('page-jump-p95', quantile(times, .95)); metric('page-jump-max', Math.max(...times))
      metric('visited-targets', jumps, 'count')
      metric('painted-page-count', pdfHost.value!.querySelectorAll('.lector-page:not(.lector-page--loading)').length, 'count')
      canvases(pdfHost.value!)
    })
    await measure(`${name}/zoom`, async () => {
      pdf.pane!.viewport.setScale(1.5)
      pdf.pane!.viewport.scrollToPage(pages - 1, false)
      await until(() => pdf.pane!.isPageReady(pages - 1), 'Zoom did not repaint the last page at current resolution', 30000)
      canvases(pdfHost.value!); await delay(2000)
    })
  } finally { await cleanup(name) }
}
async function docxCase(pages: number) {
  const name = `docx-${pages}`; kind.value = 'docx'; await nextTick()
  const trace = new StageTrace()
  // Instrument the real public WASM boundaries without changing their call
  // ordering. View spans are inclusive of nested WASM/canvas calls.
  trace.wrap(ScriptorDoc, 'openDocx', 'docx-open-parse-wasm')
  for (const method of ['relayout', 'paintPage', 'paintPageBand', 'toDocumentXml']) trace.wrap(ScriptorDoc.prototype, method, `docx-${method}-wasm`)
  for (const method of ['render', 'rebuildFrames', 'updateWindow', 'drawOverlay']) trace.wrap(ScriptorView.prototype, method, `docx-${method}-inclusive`)
  trace.wrap(CanvasRenderingContext2D.prototype, 'putImageData', 'docx-putImageData-cpu')
  try {
    await measure(`${name}/open`, async () => { await docx.open(docxHost.value!, `docx-${pages}.docx`, pages, false); await frame(); canvases(docxHost.value!) })
    await measure(`${name}/scroll-zoom`, async () => {
      const host = docxHost.value!
      for (let i = 0; i < 20; i++) { host.scrollTop = (host.scrollHeight - host.clientHeight) * i / 19; await delay(100) }
      docx.view!.setZoom(1.25); await delay(2000); canvases(host)
      metric('pages', docx.view!.pageCount(), 'count')
      assert(host.scrollTop > 0, 'DOCX scroll container did not move')
    })
  } finally { trace.finish(); await cleanup(name) }
}
async function streams(updates = 80, interaction?: (tick: number) => void) {
  const base = ('A longer conversation paragraph with **emphasis**, inline `code` and Unicode 東京.\n\n').repeat(260)
  const coldStart = performance.now()
  markdownMounts.resetStats()
  const mountTrace = new StageTrace()
  markdownMounts.trace = mountTrace.record
  markdownParser.trace = mountTrace.record
  let coldRaf = 0
  try {
    lanes.value = Array.from({ length: 4 }, (_, lane) => `## Stream ${lane + 1}\n\n${base}`)
    await nextTick()
    let first = false; let coldComplete = false
    const observeCold = () => {
      if (!first && document.querySelector('.scale-stream strong')) { first = true; metric('cold-first-rich-next-raf', performance.now() - coldStart) }
      coldComplete = [...document.querySelectorAll('.scale-stream')].every(el => el.querySelector('[data-markdown-state="rich"]') && el.querySelectorAll('strong').length === 260)
      if (coldComplete) metric('cold-complete-rich-next-raf', performance.now() - coldStart)
      else coldRaf = requestAnimationFrame(observeCold)
    }
    coldRaf = requestAnimationFrame(observeCold)
    // Keep the original 1s settle. Delaying the updates until mounting finishes
    // would hide contention and artificially improve the old mixed p99 bar.
    await delay(1000)
    const samples: number[] = []
    for (let tick = 1; tick <= updates; tick++) {
      const start = performance.now()
      interaction?.(tick)
      lanes.value = lanes.value.map((_, lane) => `## Stream ${lane + 1}\n\n${base}${'More text. '.repeat(tick)}\n\nscale-sentinel-${tick}-${lane}`)
      await nextTick()
      do { await frame(); assert(!state.requiresReload, 'Stream workload timed out') } while (![...document.querySelectorAll('.scale-stream')].every((el, lane) => el.querySelector('[data-markdown-state="rich"]') && el.textContent?.includes(`scale-sentinel-${tick}-${lane}`)))
      samples.push(performance.now() - start); await delay(16)
      if (samples[samples.length - 1] > 40 && profile.spans.length < 512) profile.spans.push({ name: 'stream-update', start, ms: samples[samples.length - 1], phase: profile.phase })
    }
    metric('stream-update-p95', quantile(samples, .95)); metric('stream-update-p99', quantile(samples, .99))
    metric('updates-per-stream', samples.length, 'count'); metric('characters-per-stream', lanes.value[0].length, 'count')
    assert(coldComplete, 'Cold content never fully mounted')
    for (const [key, value] of Object.entries(markdownMounts.stats)) if (typeof value === 'number') metric(`mount-scheduler-${key}`, value, key.endsWith('Ms') ? 'ms' : 'count')
    for (const lane of document.querySelectorAll('.scale-stream')) {
      assert(lane.querySelectorAll('strong').length === 260, 'Stream skipped or lost settled rich paragraphs')
      assert(!lane.querySelector('.pk-md__plain'), 'Plain fallback cannot pass the rich-stream benchmark')
    }
  } finally {
    cancelAnimationFrame(coldRaf)
    markdownMounts.trace = undefined; markdownParser.trace = undefined; mountTrace.finish()
    lanes.value = []; await nextTick()
  }
}
async function graphCase(nodes: number, name = `graph-${nodes}`, mixed = false, hover = false) {
  kind.value = 'graph'; await nextTick()
  try {
    await measure(`${name}/seed`, async () => {
      db = await TraverseDb.open({ numThreads: 1 })
      // Disjoint 50-node communities, two directed links per node. No N²
      // property-match seeding; identical seed feeds the full-canvas fixture.
      for (let offset = 0; offset < nodes; offset += communitySize) {
        const count = Math.min(communitySize, nodes - offset)
        const parts = Array.from({ length: count }, (_, i) => `(n${i}:Scale {i:${offset + i}})`)
        for (let i = 0; i < count; i++) for (const step of [1, 17]) parts.push(`(n${i})-[:LINK]->(n${(i + step) % count})`)
        await db.query(`CREATE ${parts.join(',')}`, null, { timeoutMs: 15000 })
      }
      const stats = await db.stats()
      assert(stats.nodes === nodes && stats.edges === nodes * 2, `Wrong database size: ${JSON.stringify(stats)}`)
      metric('stored-nodes', stats.nodes, 'count'); metric('stored-edges', stats.edges, 'count')
      metric('engine-estimated-memory', await db.estimatedMemory(), 'bytes')
    })
    await measure(`${name}/query-default-view`, async () => {
      const result = await db!.query('MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n,r,m', null, { timeoutMs: 15000 })
      const built = buildGraph(result, false)
      metric('default-view-nodes', built.nodeCount, 'count'); metric('default-view-edges', built.edgeCount, 'count')
      metric('entities-truncated', result.entities_truncated ? 1 : 0, 'boolean')
    })
    await measure(`${name}/export-reopen`, async () => {
      const bytes = await db!.exportTvdb(); metric('export-bytes', bytes.length, 'bytes')
      db!.close(); db = await TraverseDb.open({ numThreads: 1 })
      assert((await db.loadTvdb(bytes)).ok, 'Graph export did not reopen')
      const stats = await db.stats(); assert(stats.nodes === nodes && stats.edges === nodes * 2, 'Reopened graph size differs')
    })
    await measure(`${name}/full-canvas-layout`, async () => {
      graphTrace = new StageTrace(); graphTrace.webgl()
      // Deliberately bypass the product's 1024-entity view cap. This tests the
      // real GraphCanvas at full size; it is not a claim of paginated hydration.
      const full = new Graph({ type: 'directed' })
      for (let i = 0; i < nodes; i++) full.addNode(String(i), {
        x: (Math.imul(i + 1, 16807) % 2147483647) % 10000, y: (Math.imul(i + 1, 48271) % 2147483647) % 10000,
        size: 5, label: `Node ${i}`, color: '#0369A1', originalColor: '#0369A1', fgId: i, nodeType: 'Scale', properties: { i },
      })
      for (let i = 0; i < nodes; i++) for (const step of [1, 17]) full.addDirectedEdge(String(i), String(Math.floor(i / communitySize) * communitySize + (i % communitySize + step) % communitySize), { size: 2, color: '#D0D5DA', originalColor: '#D0D5DA' })
      graph.value = full
      graphTrace.graph(full)
      metric('canvas-input-nodes', full.order, 'count'); metric('canvas-input-edges', full.size, 'count')
      const start = performance.now(); await nextTick()
      await until(() => graphHost.value?.querySelector('canvas'), 'GraphCanvas did not mount', 30000)
      await frame(); metric('mount-next-raf', performance.now() - start)
      if (mixed) await streams(80, hover ? tick => {
        const renderer = graphTrace!.renderer?.deref()
        assert(renderer, 'Hover diagnostic lost its renderer')
        // A separate stress fixture, never substituted for the original bar.
        // Deliver real captor events to visible graph coordinates, including
        // direct node-to-node leave/enter transitions. No OS cursor movement.
        const point = renderer.graphToViewport(full.getNodeAttributes(String((tick * 137) % nodes)) as { x: number; y: number })
        const canvas = renderer.getCanvases().mouse, box = canvas.getBoundingClientRect()
        canvas.dispatchEvent(new MouseEvent('mousemove', { clientX: box.left + point.x, clientY: box.top + point.y, bubbles: true }))
      } : undefined)
      // Require the terminal worker frame to have reached the real renderer.
      // A mounted canvas or an arbitrary sleep is not layout completion.
      await until(() => {
        assert(!graphCanvas.value?.layoutError, graphCanvas.value?.layoutError ?? 'Layout failed')
        return graphCanvas.value?.layoutStats?.phase === 'complete'
      }, 'Full graph layout never completed', 30000)
      await frame()
      const stats = graphCanvas.value!.layoutStats!
      assert(stats.nodes === nodes && stats.edges === nodes*2, 'Layout dropped graph entities')
      for (const [key,value] of Object.entries(stats)) if (typeof value === 'number') metric(`layout-${key}`, value, key.endsWith('Ms') ? 'ms' : key.endsWith('Bytes') ? 'bytes' : 'count')
      metric('layout-stage-budget-limited', stats.reason === 'time-budget' ? 1 : 0, 'boolean')
      await delay(1000); canvases(graphHost.value!)
      graphTrace.finish(); graphTrace = null
    })
  } finally { graphTrace?.finish(); graphTrace = null; await cleanup(name) }
}
async function run(plan: string) {
  profile.events = []; profile.metrics = []; profile.spans = []
  profile.graphTriggers = []; profile.graphTriggersDropped = 0
  await cleanup('baseline', 3000)
  const one = async (name: string, body: () => Promise<void>) => {
    const ok = await check(name, async () => { await body(); return 'Completed; phase timings, frame gaps and counts are in report.profile. Native process samples are separate.' }, 240000)
    if (!ok) throw new Error(`Scale suite stopped at ${name}; inspect the report before retrying`)
  }
  try {
    if (plan === 'scale' || plan === 'scalepdf') {
      for (const pages of [100, 500, 1000]) await one(`PDF ${pages} pages`, () => pdfCase(pages))
      await one('PDF 100 unique raster images', () => pdfCase(100, true))
    }
    if (plan === 'scale' || plan === 'scaledocx') for (const pages of [50, 200]) await one(`DOCX ${pages} pages`, () => docxCase(pages))
    if (plan === 'scale' || plan === 'scalegraph') for (const nodes of [1000, 10000]) await one(`Graph ${nodes} nodes / ${nodes * 2} edges`, () => graphCase(nodes, `graph-${nodes}`, nodes === 10000))
    if (plan === 'scalegraphhover') await one('Mixed graph with synthetic node-to-node pointer movement', () => graphCase(10000, 'graph-hover-10000', true, true))
    // Destructive-to-this-lab stress runs require the external process guard;
    // they are intentionally excluded from the ordinary native Scale button.
    if (plan === 'scalegraphlimit') await one('Graph 50000 nodes / 100000 edges', () => graphCase(50000))
    if (plan === 'scalegraphrepeat') for (let cycle = 1; cycle <= 5; cycle++) await one(`Mixed graph reopen cycle ${cycle}`, async () => {
      const name = `graph-cycle-${cycle}`
      await graphCase(10000, name, true)
      await measure(`${name}/idle-10s`, () => delay(10000))
    })
    if (plan === 'scalepdfwalk') await one('PDF 1000 pages, exhaustive walk', () => pdfCase(1000, false, 'pdf-walk-1000', 1000))
    if (plan === 'scalepdfrepeat') for (let cycle = 1; cycle <= 5; cycle++) await one(`PDF-only reopen cycle ${cycle}`, async () => {
      await pdfCase(1000, false, `pdf-cycle-${cycle}`, 10)
      await measure(`pdf-cycle-${cycle}/idle-10s`, () => delay(10000))
    })
    if (plan === 'scalestream') await one('Four long streams, no graph or document', async () => {
      try { await measure('streams-only/updates', () => streams()) } finally { await cleanup('streams-only') }
    })
    if (plan === 'scalestreamlong') await one('Four long streams, 640 updates each', async () => {
      try { await measure('streams-long/updates', () => streams(640)) } finally { await cleanup('streams-long') }
    })
    if (plan === 'scalestreamrepeat') for (let cycle = 1; cycle <= 5; cycle++) await one(`Long-stream reopen cycle ${cycle}`, async () => {
      const name = `stream-cycle-${cycle}`
      try { await measure(`${name}/updates`, () => streams()) } finally { await cleanup(name) }
      await measure(`${name}/idle-10s`, () => delay(10000))
    })
    if (plan === 'scale' || plan === 'scalerepeat') {
      for (let cycle = 1; cycle <= 5; cycle++) {
        await one(`Reopen cycle ${cycle}`, async () => {
          await pdfCase(100, false, `cycle-${cycle}-pdf`, 10)
          await graphCase(10000, `cycle-${cycle}-graph`)
          await measure(`cycle-${cycle}/idle-10s`, () => delay(10000))
        })
      }
    }
  } finally { phase(state.requiresReload ? 'timed-out' : 'complete'); label.value = 'Scale run finished. Review the report and native physical-footprint samples; dimensions are not memory allocation.' }
}
defineExpose({ run })
onBeforeUnmount(() => { db?.close(); docx.close(); void pdf.close() })
</script>

<template>
  <section class="scale-lab">
    <h2>Memory &amp; scale <small>{{ label }}</small></h2>
    <p v-if="kind === 'idle'">No scale document or graph is mounted. The process sampler can measure the idle/teardown footprint.</p>
    <div v-show="kind === 'pdf'" ref="pdfHost" class="document-stage pdf-stage" />
    <div v-show="kind === 'docx'" ref="docxHost" class="document-stage docx-stage" />
    <div v-show="kind === 'graph'" ref="graphHost" class="graph-stage"><GraphCanvas v-if="graph" ref="graphCanvas" :graph="graph" :dark="false" :trace="traceGraph" export-name="scale-lab" /></div>
    <div v-if="lanes.length" class="stream-grid scale-streams">
      <div v-for="(lane, index) in lanes" :key="index" class="stream-lane scale-stream">
        <MarkdownContent :content="lane" streaming />
      </div>
    </div>
  </section>
</template>

<style scoped>
.scale-streams { position: fixed; bottom: 16px; left: 28px; width: min(650px, 60vw); z-index: 2; background: var(--bg); }
.scale-streams .stream-lane { height: 110px; }
</style>
