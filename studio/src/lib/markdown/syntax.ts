import { bundledLanguages, createHighlighter } from 'shiki'
import { MD_LANGS } from './languages'
export interface HighlightInput { code: string; language: string; dark: boolean }
export const MAX_CODE_CHARS = 128 * 1024
const CACHE_BYTES = 4 * 1024 * 1024
const aliases: Record<string, string> = { js: 'javascript', ts: 'typescript', py: 'python', rs: 'rust', sh: 'bash', shell: 'bash', yml: 'yaml', md: 'markdown', 'c++': 'cpp', 'c#': 'csharp', cs: 'csharp', rb: 'ruby' }
const allowed = new Set(MD_LANGS)
export const languageFor = (value: string) => {
  const language = value.trim().toLowerCase().split(/\s/, 1)[0]
  const normalized = aliases[language] ?? language
  return allowed.has(normalized) && normalized in bundledLanguages ? normalized : 'text'
}

export class SyntaxRenderer {
  private engine?: Awaited<ReturnType<typeof createHighlighter>>
  private cache = new Map<string, { html: string; bytes: number }>()
  get stats() { return { entries: this.cache.size, bytes: [...this.cache.values()].reduce((n, x) => n + x.bytes, 0), grammars: this.engine?.getLoadedLanguages().length ?? 0 } }
  async render(input: HighlightInput): Promise<string> {
    if (input.code.length > MAX_CODE_CHARS || input.code.split('\n').some(line => line.length > 8192)) throw new Error('Large code block shown without highlighting')
    const language = languageFor(input.language)
    const key = JSON.stringify([language, input.dark, input.code])
    const cached = this.cache.get(key)
    if (cached) { this.cache.delete(key); this.cache.set(key, cached); return cached.html }
    // Shiki grammars/regex engines retain memory. Recycle rather than loading
    // every language encountered during a long-lived chat session forever.
    if (this.engine && this.engine.getLoadedLanguages().length >= 32 && !this.engine.getLoadedLanguages().includes(language)) {
      this.engine.dispose(); this.engine = undefined
    }
    this.engine ??= await createHighlighter({ langs: [], themes: ['github-dark', 'github-light'] })
    if (language !== 'text' && !this.engine.getLoadedLanguages().includes(language)) {
      await this.engine.loadLanguage(bundledLanguages[language as keyof typeof bundledLanguages])
    }
    const html = this.engine.codeToHtml(input.code, { lang: language, theme: input.dark ? 'github-dark' : 'github-light' })
    if (html.length > 1024 * 1024) throw new Error('Highlighted output exceeds budget; showing complete plain code')
    const bytes = (key.length + html.length) * 2
    if (bytes <= CACHE_BYTES) {
      this.cache.set(key, { html, bytes })
      while (this.cache.size > 32 || this.stats.bytes > CACHE_BYTES) this.cache.delete(this.cache.keys().next().value!)
    }
    return html
  }
}
