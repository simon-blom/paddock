import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'
import { mkdir, copyFile, readFile, readdir, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { generateFixtures } from './fixtures.mjs'
import { DISPOSAL_SHIM } from './compat.mjs'

const path = (relative: string) => fileURLToPath(new URL(relative, import.meta.url))
const output = path('../../apps/macos/.build/renderer-web/')
// Lector's lowered `using` helpers and computed disposal methods must see the
// same symbols before classes evaluate, in the page and each worker realm.
// This is an additive standards shim, not eval or a WebKit security override.

// A separate entry and output: building the lab cannot replace the Studio bundle.
export default defineConfig({
  root: path('./'), base: './', publicDir: false,
  plugins: [vue(), {
    name: 'synthetic-lab-fixtures',
    async closeBundle() {
      await generateFixtures(output)
      await mkdir(`${output}/pdfium`, { recursive: true })
      for (const name of ['pdfium-st.js', 'pdfium-st.wasm']) {
        await copyFile(path(`../public/pdfium/${name}`), `${output}/pdfium/${name}`)
      }
      const lock = JSON.parse(await readFile(path('../package-lock.json'), 'utf8'))
      const dependencies = Object.fromEntries(['vue', 'markstream-vue', 'shiki', 'katex', 'mermaid', 'sigma', 'vite'].map(name => [name, lock.packages[`node_modules/${name}`].version]))
      const wasm = (await readdir(`${output}/assets`)).filter(name => name.endsWith('.wasm')).map(name => `assets/${name}`)
      wasm.push('pdfium/pdfium-st.wasm')
      const hashes = Object.fromEntries(await Promise.all(wasm.map(async name => [name, createHash('sha256').update(await readFile(`${output}/${name}`)).digest('hex')])))
      // Record the control/candidate implementation actually built, not just
      // the worktree state at profile time (source may already be restored).
      const sources = ['src/components/chat/graph/GraphCanvas.vue', 'src/lib/graph/canvas-budget.ts', 'src/lib/graph/hover-refresh.ts',
        'src/components/chat/MarkdownContent.vue', 'src/lib/markdown/mount-scheduler.ts', 'src/lib/markdown/runtime.ts',
        'src/lib/markdown/parse.ts', 'src/lib/markdown/parse.worker.ts', 'src/lib/markdown/worker-rpc.ts',
        'renderer-lab/ScaleLab.vue', 'renderer-lab/trace.ts', 'renderer-lab/graph-triggers.mjs', 'renderer-lab/graph-canvas-checks.ts']
      const sourceSHA256 = Object.fromEntries(await Promise.all(sources.map(async name => [name, createHash('sha256').update(await readFile(path(`../${name}`))).digest('hex')])))
      await writeFile(`${output}/manifest.json`, JSON.stringify({ builtAt: new Date().toISOString(), dependencies, wasmSHA256: hashes, sourceSHA256 }, null, 2))
    },
  }],
  resolve: { alias: {
    '@': path('../src'),
    '@truespar/scriptor-vue': path('../vendor/scriptor/vue/src/index.ts'),
    '@truespar/scriptor-core': path('../vendor/scriptor/core/src/index.ts'),
    '@truespar/scriptor-wasm': path('../vendor/scriptor/scriptor-wasm/dist/scriptor_wasm.js'),
  } },
  worker: { format: 'es', rolldownOptions: { output: { banner: DISPOSAL_SHIM } } }, assetsInclude: ['**/*.wasm'],
  build: { outDir: output, emptyOutDir: true, target: 'safari18', chunkSizeWarningLimit: 2500,
    rolldownOptions: { output: { banner: DISPOSAL_SHIM } },
  },
})
