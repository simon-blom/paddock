import { setCustomComponents } from 'markstream-vue'
import MarkdownCode from '@/components/chat/MarkdownCode.vue'
import MarkdownMap from '@/components/chat/MarkdownMap.vue'
import { initMarkstream } from './markstream'
let initialized = false
export function initMarkdownMaps() {
  initMarkstream()
  if (initialized) return
  initialized = true
  // Extend the parsed fence language, never split the document string. Late
  // references still reach earlier paragraphs, and a literal ```map inside a
  // longer example fence never becomes a map. Maps keep their existing offline
  // default and explicit opt-in for remote tiles in PhotoLocation.
  setCustomComponents('paddock-rich-maps', { code_block: MarkdownCode, map: MarkdownMap })
}
