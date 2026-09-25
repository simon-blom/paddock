// Guard both entrypoints, workers and vendored viewer code at every build.
// A dependency refresh must not silently reintroduce origin-local persistence.
import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
const root = fileURLToPath(new URL('..', import.meta.url))
function files(path) {
  if (statSync(path).isDirectory()) return readdirSync(path).flatMap(name => files(join(path, name)))
  return /\.(?:[cm]?js|ts|vue|html)$/.test(path) && !/\.(?:test|d)\.ts$/.test(path) ? [path] : []
}
const violations = []
const forbidden = /\b(?:localStorage|sessionStorage|indexedDB|IDBFactory|IDBDatabase|openDatabase)\b|\bcaches\s*\.\s*(?:open|match)|\bstorage\s*\.\s*(?:getDirectory|persist)|\bdocument\s*\.\s*cookie\s*=/
for (const path of ['src', 'native-workspace', 'native-renderer', 'renderer-lab', 'vendor', 'public', 'index.html'].flatMap(dir => files(join(root, dir)))) {
  const name = relative(root, path).replaceAll('\\', '/')
  // Read/delete-only legacy migration is the one exception, not a fallback.
  if (name === 'src/lib/browser-storage-migration.ts') {
    if (/\.setItem\s*\(|\.clear\s*\(/.test(readFileSync(path, 'utf8'))) violations.push(name)
    continue
  }
  const text = readFileSync(path, 'utf8').replace(/\/\*[\s\S]*?\*\/|^\s*\/\/.*$/gm, '')
  if (forbidden.test(text)) violations.push(name)
}
if (violations.length) throw new Error(`Browser persistence is forbidden; use Rust/SQLite:\n${violations.join('\n')}`)

// Check emitted dependencies as well as our sources. The entry containing the
// legacy importer may contain its two read-only probes; no other chunk may
// acquire browser persistence, including PDF and graph worker assets.
export function browserStorageBundleGuard() {
  return {
    name: 'paddock-no-browser-persistence',
    generateBundle(_options, bundle) {
      for (const [name, chunk] of Object.entries(bundle)) {
        if (!/\.(?:js|html)$/.test(name)) continue
        const code = chunk.type === 'chunk' ? chunk.code : String(chunk.source)
        const migration = chunk.type === 'chunk' && Object.keys(chunk.modules).some(id => id.endsWith('/src/lib/browser-storage-migration.ts'))
        const reads = [...code.matchAll(/\blocalStorage\b/g)].length
        if (reads > (migration ? 2 : 0)
          || /\b(?:sessionStorage|indexedDB|IDBFactory|IDBDatabase|openDatabase)\b|\bcaches\s*\.\s*(?:open|match)|\bstorage\s*\.\s*(?:getDirectory|persist)|\bdocument\s*\.\s*cookie\s*=/.test(code)) {
          this.error(`Browser persistence found in emitted asset ${name}`)
        }
      }
    },
  }
}
