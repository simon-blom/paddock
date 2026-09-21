// Streaming Markdown renderer config (markstream-vue + Shiki + KaTeX + Mermaid).
import { enableKatex, enableMermaid, setCustomComponents } from 'markstream-vue'
import MarkdownCode from '@/components/chat/MarkdownCode.vue'
import 'markstream-vue/index.css'
import 'katex/dist/katex.min.css'
// Our dark-theme overrides for markstream's hardcoded-light blocks (after its
// stylesheet so they win).
import '@/styles/markstream-overrides.css'

export { MD_LANGS } from './markdown/languages'

let inited = false

/** Enable math + Mermaid once. Mermaid is a lazy bundled chunk; its SVG/DOM
 *  rendering remains owned by Markstream. Markdown parsing and syntax
 *  tokenization use our separate workers in lib/markdown. */
export function initMarkstream(): void {
  if (inited) return
  inited = true
  setCustomComponents('paddock-rich', { code_block: MarkdownCode })
  enableKatex()
  enableMermaid(() => import('mermaid'))
}
