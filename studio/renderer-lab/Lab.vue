<script setup lang="ts">
import { nextTick, onMounted, onBeforeUnmount, ref, shallowRef } from 'vue'
import MarkdownContent from '@/components/chat/MarkdownContent.vue'
import GraphCanvas from '@/components/chat/graph/GraphCanvas.vue'
import { initMarkstream } from '@/lib/markstream'
import { TraverseDb } from '@truespar/traverse-wasm'
import { buildGraph } from '@/lib/graph/session'
import type Graph from 'graphology'
import { TooltipProvider } from 'reka-ui'
import { PDFCase, DOCXCase } from './documents'
import ScaleLab from './ScaleLab.vue'
import { markdownChecks } from './markdown-checks'
import { graphCanvasChecks } from './graph-canvas-checks'
import MarkdownScrollCase from './MarkdownScrollCase.vue'
import { markdownParser, syntaxHighlighter } from '@/lib/markdown/runtime'
import { markdownMounts } from '@/lib/markdown/mount-scheduler'
import { assert, check, delay, manual, publish, reportError, state, until, type Command } from './report'

initMarkstream()
const tabs = ['markdown', 'math', 'mermaid', 'pdf', 'docx', 'traverse', 'stress', 'security'] as const
const tab = ref<(typeof tabs)[number] | 'scale'>('markdown')
const scaleLab = ref<InstanceType<typeof ScaleLab>>()
const dark = ref(false)
const content = ref('Choose **Run all** to exercise the bundled renderers, or select a tab and run that fixture.')
const streaming = ref(false)
const markdownHost = ref<HTMLElement>()
const scrollCase = ref<InstanceType<typeof MarkdownScrollCase>>()
const checkingScroll = ref(false)
const pdfHost = ref<HTMLElement>()
const docxHost = ref<HTMLElement>()
const graph = shallowRef<Graph | null>(null)
const lanes = ref<string[]>([])
const pdf = new PDFCase()
const docx = new DOCXCase()
let db: TraverseDb | null = null
let exported: Uint8Array | null = null
const note = ref('This is a renderer compatibility lab, not the Studio redesign.')

const prose = `# Streaming renderer\n\nA **bold** statement, *emphasis*, ~~revision~~, and inline \`code\`. Unicode: naïve · 東京 · Ελληνικά · 🐾.\n\n- First item\n- Second item\n  - Nested item\n\n> A blockquote should preserve its indentation.\n\n| Engine | Runtime |\n| --- | --- |\n| Traverse | WASM worker |\n| Scriptor | WASM canvas |\n\n\`\`\`swift\nstruct Reply: Sendable {\n    let text: String\n}\n\`\`\`\n\nEnd of stream sentinel.\n`
const math = String.raw`# Mathematics

Inline $E = mc^2$ and $\alpha + \beta = \gamma$.

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}
$$

$$
\begin{aligned}
\nabla \cdot \mathbf{E} &= \frac{\rho}{\varepsilon_0} \\
\nabla \times \mathbf{B} &= \mu_0\mathbf{J} + \mu_0\varepsilon_0\frac{\partial\mathbf{E}}{\partial t}
\end{aligned}
$$

$$
A = \begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix},\quad P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}
$$
`
const mermaid = '# Mermaid diagram\n\n```mermaid\nflowchart LR\n  Swift[Swift shell] --> WK[WebKit renderer]\n  WK --> Math[KaTeX and Mermaid]\n  WK --> PDF[Lector PDFium WASM]\n  WK --> Word[Scriptor WASM]\n  WK --> Graph[Traverse worker]\n```\n\nThe diagram above must be SVG, not a highlighted code fence.'

async function choose(name: (typeof tabs)[number] | 'scale') {
  tab.value = name
  if (name === 'markdown') content.value = prose
  if (name === 'math') content.value = math
  if (name === 'mermaid') content.value = mermaid
  await nextTick()
  await delay(80)
}
async function workerProbe() {
  await check('Module worker + WASM', async () => {
    const worker = new Worker(new URL('./probe.worker.ts', import.meta.url), { type: 'module' })
    try {
      const result = await new Promise<{ wasm: boolean; module: string }>((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Module worker did not respond')), 5000)
        worker.onmessage = event => { clearTimeout(timer); resolve(event.data) }
        worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message || 'Worker failed to load')) }
      })
      assert(result.wasm && result.module.includes('probe.worker'), 'Worker did not instantiate WASM')
      return `WASM instantiated in bundled module worker: ${result.module}`
    } finally { worker.terminate() }
  })
}
async function runMarkdown() {
  await choose('markdown')
  await check('Markdown streaming + highlighting', async () => {
    streaming.value = true; content.value = ''
    for (let i = 0; i < prose.length; i += 13) {
      content.value = prose.slice(0, i + 13)
      await delay(12)
    }
    streaming.value = false
    await until(() => markdownHost.value?.textContent?.includes('End of stream sentinel.'), 'Final streamed text missing')
    await until(() => markdownHost.value?.querySelector('table') && markdownHost.value.querySelector('pre span[style*="color"]'), 'Table or Shiki token colors missing')
    assert(markdownHost.value?.querySelector('strong') && markdownHost.value.querySelector('blockquote'), 'Markdown semantics missing')
    return `${prose.length} characters in 13-character chunks; final sentinel, table, emphasis, quote and highlighted Swift verified`
  })
  await markdownChecks(content, streaming, dark, markdownHost)
  checkingScroll.value = true; await nextTick()
  try { await scrollCase.value!.run() } finally { checkingScroll.value = false; await nextTick() }
  await check('Production Markdown transcript admission', async () => {
    const before = markdownParser.stats.owners
    lanes.value = Array.from({ length: 48 }, (_, i) => `**History ${i}**\n\n\`\`\`swift\nlet message = ${i}\n\`\`\`\n`)
    streaming.value = false
    try {
      await nextTick()
      await until(() => {
        const blocks = [...document.querySelectorAll('.stream-grid .stream-lane')]
        return blocks.length === 48 && blocks.every((block, i) => block.querySelector('[data-markdown-state="rich"]')
          && block.querySelector('strong')?.textContent === `History ${i}` && block.querySelector('.pk-code pre span[style*="color"]'))
      }, 'Transcript beyond the work-queue limit did not recover rich rendering', 20000)
      assert(markdownParser.stats.queued === 0 && markdownParser.stats.waiting === 0, 'Parser admission did not drain')
      assert(syntaxHighlighter.stats.queued === 0 && syntaxHighlighter.stats.waiting === 0, 'Syntax admission did not drain')
    } finally { lanes.value = []; await nextTick() }
    assert(markdownParser.stats.owners === before, 'Transcript renderer owners leaked after unmount')
    return '48 simultaneous rich messages and Swift blocks exceed the 32-job queue; all eventually render, queues drain, and unmount releases every test owner'
  })
}
async function runMath() {
  await choose('math')
  await check('KaTeX inline + display + matrices', async () => {
    content.value = math; streaming.value = false
    await until(() => (markdownHost.value?.querySelectorAll('.katex').length ?? 0) >= 5, 'Expected 5 KaTeX expressions')
    assert(!markdownHost.value?.querySelector('.katex-error'), 'KaTeX parse error')
    await document.fonts.ready
    return `${markdownHost.value!.querySelectorAll('.katex').length} expressions rendered; bundled fonts ready; no KaTeX error nodes`
  })
}
async function runMermaid() {
  await choose('mermaid')
  await check('Mermaid SVG diagram', async () => {
    content.value = mermaid; streaming.value = false
    await until(() => [...(markdownHost.value?.querySelectorAll('svg') ?? [])].some(svg => svg.textContent?.includes('Traverse worker')), 'Mermaid never produced the expected SVG')
    return 'Bundled Mermaid produced SVG with the expected graph labels'
  })
}
async function runPDF() {
  await choose('pdf')
  if (await check('Lector PDF render', () => pdf.open(pdfHost.value!))) {
    await check('PDF full-text search', () => pdf.search())
    await check('PDF OCR overlay', () => pdf.overlay(pdfHost.value!))
  }
}
async function runDOCX() {
  await choose('docx')
  if (await check('Scriptor DOCX render', () => docx.open(docxHost.value!))) {
    await check('DOCX tracked changes + zoom', () => docx.review())
  }
}
async function runTraverse() {
  await choose('traverse')
  await graphCanvasChecks()
  await check('Traverse query + export/reopen', async () => {
    db?.close(); db = null; graph.value = null
    db = await TraverseDb.open({ numThreads: 1 })
    await db.query("CREATE (a:Engine {name:'Swift'}), (b:Engine {name:'WebKit'}), (c:Engine {name:'WASM'}), (a)-[:HOSTS]->(b), (b)-[:RUNS]->(c)")
    const stats = await db.stats()
    assert(stats.nodes === 3 && stats.edges === 2, `Wrong graph stats: ${JSON.stringify(stats)}`)
    const result = await db.query('MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n,r,m')
    const built = buildGraph(result, dark.value)
    assert(built.nodeCount === 3 && built.edgeCount === 2, 'Hydrated graph has missing nodes/edges')
    graph.value = built.graph
    exported = await db.exportTvdb()
    assert(exported.byteLength > 100, 'Empty Traverse export')
    db.close(); db = await TraverseDb.open({ numThreads: 1 })
    const loaded = await db.loadTvdb(exported.slice())
    assert(loaded.ok, 'Exported database did not reopen')
    const reopened = await db.stats()
    assert(reopened.nodes === 3 && reopened.edges === 2, 'Reopened graph differs')
    await nextTick()
    await until(() => document.querySelector('.graph-stage canvas'), 'Studio graph canvas did not mount')
    await delay(5500)
    return `Traverse ${db.version}: 3 nodes / 2 edges, ${exported.length} export bytes; identical reopened stats; Studio Sigma canvas mounted (visual check required)`
  })
  await check('Traverse bounded results + cancellation recovery', async () => {
    assert(db, 'Open Traverse first')
    const rows = await db.query('UNWIND range(1,12000) AS i RETURN i')
    assert(rows.rows.length === 10000 && rows.total_rows === 12000 && rows.truncated, 'Row cap lost total count or truncation')
    const large = await db.query('RETURN $text AS text', {text:'x'.repeat(3*1024*1024)})
    assert(large.rows.length === 0 && large.total_rows === 1 && large.truncated, 'Oversized cell crossed the conversion budget')
    const before = await db.stats(), cancel = new AbortController()
    const active = db.query('CREATE (:CallerCancelled)', null, {signal:cancel.signal})
    const cancelled = active.then(() => false, (e: Error & {executionMayHaveCompleted?:boolean}) => e.name === 'AbortError' && e.executionMayHaveCompleted === true)
    cancel.abort()
    assert(await cancelled, 'Active cancellation falsely promised rollback')
    assert((await db.stats()).nodes === before.nodes+1, 'Active cancellation killed or lost the database')
    assert(db.queueStats.active === 0 && db.queueStats.queued === 0 && db.queueStats.bytes === 0, 'Physical request slot leaked')
    let rejected = false
    try { await db.query('UNWIND range(1,1000000) AS i RETURN i', null, {timeoutMs:1}) }
    catch (e) { rejected = /timed out|memory|budget/i.test(String(e)) }
    assert(rejected, 'Oversized query did not enforce cooperative limits')
    assert((await db.query('RETURN 7 AS value')).rows[0]?.[0] === 7, 'Query limits poisoned the next request')
    await db.query('MATCH (n:CallerCancelled) DETACH DELETE n')
    return 'Row/byte caps, explicit caller-only cancellation, retained mutable state and post-limit recovery verified in the real WebKit worker.'
  })
}
function percentile(samples: number[], p: number) {
  return [...samples].sort((a, b) => a - b)[Math.min(samples.length - 1, Math.ceil(samples.length * p) - 1)] ?? 0
}
async function runStress() {
  await choose('stress')
  await check('Four streams + document activity', async () => {
    assert(pdf.handle && docx.view && db, 'Run PDF, DOCX and Traverse successfully before stress')
    lanes.value = ['', '', '', '']; streaming.value = true
    const frames: number[] = []; const updates: number[] = []
    let last = performance.now(); let live = true
    const frame = () => { const now = performance.now(); frames.push(now - last); last = now; if (live) requestAnimationFrame(frame) }
    requestAnimationFrame(frame)
    try {
      for (let tick = 1; tick <= 160; tick++) {
        const start = performance.now()
        // Every tick changes all four streams; do not benchmark repeated writes
        // of an already-finished short fixture (Vue correctly skips those).
        lanes.value = lanes.value.map((_, lane) => `### Stream ${lane + 1}\n\n${prose.repeat(4).slice(0, tick * 9)}\n\nstream-sentinel-${tick}-${lane}`)
        await nextTick()
        const arrived = () => [...document.querySelectorAll('.stream-lane')].every((el, lane) => el.textContent?.includes(`stream-sentinel-${tick}-${lane}`))
        do {
          await new Promise<void>(resolve => requestAnimationFrame(() => resolve()))
          assert(!state.requiresReload, 'Timed-out stream test requires reload')
        } while (!arrived())
        updates.push(performance.now() - start)
        if (tick % 40 === 0) {
          dark.value = !dark.value
          document.documentElement.classList.toggle('dark', dark.value)
          docx.view!.setZoom(tick % 80 ? 0.7 : 0.75)
          await pdf.search()
          const stats = await db!.stats(); assert(stats.nodes === 3, 'Graph failed during mixed workload')
          const canvas = pdf.pane!.canvas
          canvas.scrollTop = tick % 80 ? 400 : 0
        }
        await delay(12)
      }
    } finally { live = false; streaming.value = false }
    const p95 = percentile(updates, .95); const p99 = percentile(updates, .99)
    const detail = `160 distinct updates × 4 streams; update-to-DOM-sentinel/next-rAF proxy p95=${p95.toFixed(1)} ms, p99=${p99.toFixed(1)} ms; frame gap p99=${percentile(frames, .99).toFixed(1)} ms, max=${Math.max(...frames).toFixed(1)} ms. Not presentation latency or inference throughput.`
    assert(p95 <= 50 && p99 <= 100, `Responsiveness target missed: ${detail}`)
    return detail
  }, 30000)
  if (state.requiresReload) return
  await check('Document teardown + reopen', async () => {
    await choose('pdf'); await pdf.open(pdfHost.value!); await pdf.search()
    await choose('docx'); await docx.open(docxHost.value!); await docx.review()
    await choose('stress')
    return 'PDF engine/worker and DOCX view destroyed and recreated; page counts, ink and search revalidated (not a leak test)'
  }, 30000)
}
async function runSecurity() {
  await choose('security')
  await check('CSP denies remote fetch', async () => {
    let blocked = false
    const handler = (e: SecurityPolicyViolationEvent) => { if (e.effectiveDirective === 'connect-src') blocked = true }
    document.addEventListener('securitypolicyviolation', handler)
    try {
      let succeeded = false
      try { await fetch('https://renderer-lab.invalid/never-send'); succeeded = true } catch { /* expected */ }
      await delay(100)
      assert(blocked && !succeeded, 'Fetch failed without a confirmed CSP block')
      return 'Remote fetch was rejected by connect-src; no network backend is needed'
    } finally { document.removeEventListener('securitypolicyviolation', handler) }
  })
  await check('Inert artifact frame', async () => {
    const frame = document.createElement('iframe')
    frame.sandbox.value = ''
    frame.title = 'Sandboxed synthetic artifact'
    const blob = URL.createObjectURL(new Blob(['<h3>Inert artifact</h3><p>Scripts, forms and top navigation have no sandbox grants.</p><script>parent.postMessage("lab-frame-script-ran", "*")<\/script>'], { type: 'text/html' }))
    frame.src = blob
    let ran = false; const handler = (e: MessageEvent) => { if (e.data === 'lab-frame-script-ran') ran = true }
    window.addEventListener('message', handler)
    document.querySelector('#security-fixture')!.replaceChildren(frame)
    try {
      await delay(500)
      assert(!ran && frame.sandbox.length === 0, 'Artifact executed script')
      let opaque = false
      try { opaque = !frame.contentDocument } catch { opaque = true }
      assert(opaque, 'Artifact unexpectedly shares parent DOM')
      return 'Opaque sandbox frame has no script, same-origin, popup, form or top-navigation grants; no script message received'
    } finally { window.removeEventListener('message', handler); URL.revokeObjectURL(blob) }
  })
  manual('Security qualification', 'Smoke checks only. Production still needs Markdown/link sanitization fuzzing, bounded attachment decoding, bridge audit, denial-of-service limits and process-recovery tests.')
}

const cases = { markdown: runMarkdown, math: runMath, mermaid: runMermaid, pdf: runPDF, docx: runDOCX, traverse: runTraverse, stress: runStress, security: runSecurity }
async function command(name: Command) {
  if (state.requiresReload) { note.value = 'A test timed out. Use native Reload before running more fixtures.'; return }
  if (name === 'theme') {
    dark.value = !dark.value; document.documentElement.classList.toggle('dark', dark.value); return
  }
  if (state.running) return
  if (name === 'reset') {
    await pdf.close(); docx.close(); db?.close(); db = null; graph.value = null; state.checks = []; publish(); return
  }
  state.running = true; publish()
  try {
    if (name === 'all') {
      state.checks = []
      await workerProbe()
      for (const item of tabs) {
        await cases[item](); publish(item); await delay(300)
        if (state.requiresReload) break
      }
    } else if (name.startsWith('scale')) {
      await pdf.close(); docx.close(); db?.close(); db = null; graph.value = null; exported = null; lanes.value = []; content.value = ''
      await choose('scale'); await scaleLab.value!.run(name)
    } else { await cases[name as keyof typeof cases](); publish(name) }
    manual('Interaction and accessibility', 'Check physical selection/copy, keyboard Tab and Command-F, PDF OCR hit targets, graph pan/zoom/picking, VoiceOver and light/dark contrast. Programmatic clicks do not certify these.')
    manual('Memory and production performance', 'Inspect process footprint over repeated reopen cycles; test a large corpus and long transcript. This synthetic run does not establish a leak bound, presented-frame latency or inference overhead.')
    note.value = 'Run complete. Inspect every failed/manual result before choosing the production architecture.'
    if (name === 'all') await choose('markdown')
  } finally { state.running = false; publish() }
}
onMounted(async () => {
  window.paddockLab = { command, memoryStats: () => ({
    parser: markdownParser.stats, highlighter: syntaxHighlighter.stats, mounts: markdownMounts.stats,
    connectedElements: document.querySelectorAll('*').length,
    connectedText: (() => { let count = 0; const walker = document.createTreeWalker(document, NodeFilter.SHOW_TEXT); while (walker.nextNode()) count++; return count })(),
    scaleCanvases: document.querySelectorAll('.scale-lab canvas').length,
    scaleLanes: document.querySelectorAll('.scale-stream').length,
  }) }
  try {
    const manifest = await (await fetch('./manifest.json')).json()
    Object.assign(state.environment, { build: manifest.builtAt, dependencyVersions: JSON.stringify(manifest.dependencies), wasmSHA256: JSON.stringify(manifest.wasmSHA256), sourceSHA256: JSON.stringify(manifest.sourceSHA256 ?? {}) })
  } catch (error) { reportError('Bundle manifest', error) }
  publish()
})
onBeforeUnmount(() => { void pdf.close(); docx.close(); db?.close() })
</script>

<template>
  <TooltipProvider>
  <div class="lab">
    <nav aria-label="Renderer fixtures">
      <button v-for="name in tabs" :key="name" :aria-current="tab === name ? 'page' : undefined" :disabled="state.running" @click="choose(name)">{{ name === 'docx' ? 'Scriptor' : name === 'pdf' ? 'Lector' : name }}</button>
      <button class="run-tab" :disabled="state.running" @click="command(tab)">Run fixture</button>
      <button :disabled="state.running" :aria-current="tab === 'scale' ? 'page' : undefined" @click="choose('scale')">Scale</button>
    </nav>
    <div class="lab-body">
      <main>
        <p class="notice">{{ note }}</p>
        <ScaleLab v-if="tab === 'scale'" ref="scaleLab" />
        <MarkdownScrollCase v-if="checkingScroll" ref="scrollCase" />
        <div v-show="['markdown', 'math', 'mermaid'].includes(tab)" ref="markdownHost" class="markdown-stage">
          <MarkdownContent :content="content" :streaming="streaming" :is-dark="dark" @error="reportError('Production Markdown', $event)" />
        </div>
        <section v-show="tab === 'pdf' || tab === 'stress'" :class="['document-section', { compact: tab === 'stress' }]">
          <h2>Lector · PDFium WASM <small>2 pages · search "Labrador" · selectable text · OCR overlay</small></h2>
          <div ref="pdfHost" class="document-stage pdf-stage" />
        </section>
        <section v-show="tab === 'docx' || tab === 'stress'" :class="['document-section', { compact: tab === 'stress' }]">
          <h2>Scriptor · OOXML WASM <small>2 pages · table · image · tracked changes</small></h2>
          <div ref="docxHost" class="document-stage docx-stage" />
        </section>
        <section v-show="tab === 'traverse'" class="graph-stage">
          <h2>Traverse · query, layout, export and reopen</h2>
          <GraphCanvas v-if="graph" :graph="graph" :dark="dark" export-name="renderer-lab" />
        </section>
        <section v-show="tab === 'stress'" class="stream-grid">
          <div v-for="(lane, index) in lanes" :key="index" class="stream-lane">
            <MarkdownContent :content="lane" :streaming="streaming" :is-dark="dark" @error="reportError('Production Markdown stream', $event)" />
          </div>
        </section>
        <section v-show="tab === 'security'" class="security-stage">
          <h2>Offline and trust boundary</h2>
          <p>Only bundled assets are available. The native bridge accepts bounded diagnostic reports, not commands, credentials or arbitrary file paths.</p>
          <div id="security-fixture" />
          <p><a href="https://renderer-lab.invalid/navigation-test">Blocked remote navigation fixture</a></p>
        </section>
      </main>
      <aside aria-label="Diagnostic results" aria-live="polite">
        <h2>Results <small>{{ state.running ? 'Running…' : 'Bundled WebKit' }}</small></h2>
        <article v-for="result in state.checks" :key="result.name" :class="['result', result.status]">
          <div class="result-heading"><span class="status">{{ result.status }}</span><time>{{ result.ms ? `${result.ms.toFixed(0)} ms` : '' }}</time></div>
          <h3>{{ result.name }}</h3><p>{{ result.detail }}</p>
        </article>
        <details><summary>Environment</summary><dl><template v-for="(value, key) in state.environment" :key="key"><dt>{{ key }}</dt><dd>{{ value }}</dd></template></dl></details>
      </aside>
    </div>
  </div>
  </TooltipProvider>
</template>
