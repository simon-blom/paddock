import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'
import { readFile, writeFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { DISPOSAL_SHIM } from '../build/webkit-compat.mjs'
const path = (relative: string) => fileURLToPath(new URL(relative, import.meta.url))
// Only the shared rich renderer and this narrow presentation entry are bundled.
// No router, model-management stores, fixture generator or lab diagnostics.
export default defineConfig({
  root: path('./'), base: './', publicDir: false, plugins: [vue(), {
    name: 'native-renderer-provenance',
    async closeBundle() {
      const lock = JSON.parse(await readFile(path('../package-lock.json'), 'utf8'))
      const dependencies = Object.fromEntries(['vue', 'markstream-vue', 'shiki', 'katex', 'mermaid', 'vite'].map(name => [name, lock.packages[`node_modules/${name}`].version]))
      const sources = ['native-renderer/main.ts', 'native-renderer/Transcript.vue', 'native-renderer/style.css',
        'src/components/chat/MarkdownContent.vue', 'src/components/chat/MarkdownCode.vue',
        'src/lib/markdown/mount-scheduler.ts', 'src/lib/markdown/runtime.ts', 'src/lib/markdown/parse.ts',
        'src/lib/markdown/worker-rpc.ts', 'build/webkit-compat.mjs']
      const sourceSHA256 = Object.fromEntries(await Promise.all(sources.map(async name => [name,
        createHash('sha256').update(await readFile(path(`../${name}`))).digest('hex')])))
      await writeFile(path('../../apps/macos/.build/studio-renderer/manifest.json'), JSON.stringify({ builtAt: new Date().toISOString(), dependencies, sourceSHA256 }, null, 2))
    },
  }],
  resolve: { alias: { '@': path('../src') } },
  worker: { format: 'es', rolldownOptions: { output: { banner: DISPOSAL_SHIM } } },
  build: { outDir: path('../../apps/macos/.build/studio-renderer'), emptyOutDir: true, target: 'safari18',
    rolldownOptions: { output: { banner: DISPOSAL_SHIM } } },
})
