import { defineConfig, mergeConfig } from 'vite'
import { fileURLToPath } from 'node:url'
import { readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import base from '../vite.config'
import { DISPOSAL_SHIM } from '../build/webkit-compat.mjs'
const path = (name: string) => fileURLToPath(new URL(name, import.meta.url))
export default mergeConfig(base, defineConfig({
  root: path('./'), publicDir: false, base: '/',
  plugins: [{
    name: 'native-content-provenance',
    generateBundle(_, bundle) {
      // Only Lector's engine assets. In particular, never copy the web
      // Studio's microphone worklet into the native application bundle.
      for (const file of ['pdfium-st.js', 'pdfium-st.wasm']) {
        this.emitFile({ type: 'asset', fileName: `pdfium/${file}`, source: readFileSync(path(`../public/pdfium/${file}`)) })
      }
      const forbidden = /\/components\/(?:chat\/(?:ChatView|ChatThread|MessageBubble|Composer|AudioPlayer|TranscriptView|ToolCall|WebSearchCall|ArtifactPanel|ArtifactPane)|ui\/Toaster)\.vue(?:\?|$)/
      const modules = Object.values(bundle).flatMap(chunk => chunk.type === 'chunk' ? Object.keys(chunk.modules) : [])
      if (!modules.length || !modules.some(id => id.includes('/native-workspace/ViewerOnly.vue'))) {
        this.error('Native UI boundary audit did not receive the application module graph')
      }
      const forbiddenRuntime = /\/(?:native-workspace\/(?:controller|main|audio|native-transcript)|src\/stores\/chat|src\/composables\/(?:useChatStream|useConversationEffects|useRecorder|useMicTranscribe|useLiveTurn))\.(?:ts|vue)(?:\?|$)/
      const leaked = modules.filter(id => forbidden.test(id) || forbiddenRuntime.test(id))
      if (leaked.length) this.error(`Web application UI entered the native bundle:\n${leaked.join('\n')}`)
      const sources = ['native-workspace/ViewerOnly.vue', 'native-workspace/viewer-main.ts',
        'src/components/chat/DocumentPane.vue', 'src/components/chat/graph/GraphCanvas.vue',
        'native-workspace/style.css']
      const sourceSHA256 = Object.fromEntries(sources.map(name => [name, createHash('sha256').update(readFileSync(path(`../${name}`))).digest('hex')]))
      this.emitFile({ type: 'asset', fileName: 'manifest.json', source: JSON.stringify({ surface: 'native-embedded-viewers', renderer: 'native', auditedModules: modules.length, webUI: ['Lector', 'Scriptor', 'Traverse'], forbiddenWebUI: leaked, builtAt: new Date().toISOString(), sourceSHA256, wasm: [...Object.keys(bundle).filter(n => n.endsWith('.wasm')), 'pdfium/pdfium-st.wasm'] }, null, 2) })
    },
  }],
  worker: { format: 'es', rolldownOptions: { output: { banner: DISPOSAL_SHIM } } },
  build: { outDir: path('../../apps/macos/.build/studio-workspace'), emptyOutDir: true, target: 'safari18',
    rolldownOptions: { output: { banner: DISPOSAL_SHIM } } },
}))
