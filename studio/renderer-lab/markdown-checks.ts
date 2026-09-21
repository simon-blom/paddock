import { nextTick, type Ref } from 'vue'
import { assert, check, delay, until } from './report'
import { markdownParser, syntaxHighlighter } from '@/lib/markdown/runtime'

const fixtures = [
  '# Rich text\n\nA **bold**, *italic*, ~~deleted~~ reply 東京 🐾.\n\n> A quote\n> - A nested list\n\n1. First\n2. Second\n',
  '| Key | Value |\n| --- | --- |\n| a | **one** |\n| b | `two` |\n',
  'Forward [reference][target] and a note[^n].\n\n' + 'Settled **paragraph**.\n\n'.repeat(38) + '[target]: https://example.com\n[^n]: Footnote text.\n',
  '```typescript\nconst html = "<script>alert(1)</script>"\n```\n\n~~~~markdown\n```map\n1,2\n```\n~~~~\n',
  'Unclosed **marker and [unfinished link](https://example.com\n',
  '```diff typescript\n-const oldValue = 1\n+const newValue = 2\n```\n',
]
const snapshot = (host: HTMLElement) => [...host.querySelectorAll('h1,h2,p,ul,ol,li,blockquote,table,tr,th,td,strong,em,del,pre,code,a,sup')]
  .map(el => [el.tagName, el.textContent, el.getAttribute('href')])

export async function markdownChecks(content: Ref<string>, streaming: Ref<boolean>, dark: Ref<boolean>, host: Ref<HTMLElement | undefined>) {
  const ready = () => host.value?.querySelector('[data-markdown-state="rich"]') as HTMLElement | null
  const load = async (text: string, stream: boolean) => {
    // Empty commits clear the previous DOM, so this cannot pass on a stale
    // sentinel or a previous final render while the worker is still pending.
    content.value = ''; streaming.value = false; await nextTick(); await delay(10)
    streaming.value = stream
    if (stream) {
      for (let i = 0, turn = 0; i < text.length; turn++) {
        i += [1, 7, 31][turn % 3]
        content.value = text.slice(0, i); await delay(2)
      }
      streaming.value = false
    } else content.value = text
    await nextTick()
    await until(() => ready()?.dataset.markdownFinal === 'true' && ready()?.textContent?.includes('parity-end'), 'Production Markdown final commit missing')
    if (text.includes('```typescript')) await until(() => host.value?.querySelector('.pk-code pre span[style*="color"]'), 'Worker Shiki output missing')
  }
  await check('Production Markdown final DOM parity', async () => {
    for (const fixture of fixtures) {
      const text = fixture + '\n\nparity-end'
      await load(text, true)
      const streamed = snapshot(host.value!)
      await load(text, false)
      assert(JSON.stringify(snapshot(host.value!)) === JSON.stringify(streamed), 'Streamed semantic DOM differs from one-shot DOM')
      assert(!host.value!.querySelector('.pk-md__plain'), 'Plain fallback cannot pass parity')
      if (fixture.includes('[target]:')) {
        assert(host.value!.querySelector('a[href="https://example.com"]'), 'Late reference did not update the first render group')
        assert(host.value!.textContent?.includes('Footnote text.'), 'Footnote missing across render groups')
      }
      if (fixture.includes('```diff')) assert(host.value!.querySelector('pre')?.textContent === '-const oldValue = 1\n+const newValue = 2\n', 'Diff removed lines were lost')
    }
    return '6 streamed/one-shot semantic DOM pairs; partial fences, nested lists, tables, Unicode, late references across 32-node groups, footnotes and full unified diffs'
  }, 30000)
  await check('Production Markdown HTML and URL boundary', async () => {
    await load('<script>window.__unsafeMarkdown=1</script>\n\n<img src=x onerror="window.__unsafeMarkdown=2">\n\n[unsafe](javascript:alert(1))\n\nparity-end', false)
    assert(!host.value!.querySelector('script,iframe,[onerror],[onclick],a[href^="javascript:"]'), 'Unsafe model markup reached active DOM')
    assert(host.value!.textContent?.includes('<script>'), 'Escaped HTML was silently dropped')
    return 'HTML is escaped, unsafe links are inert, and model script/handler nodes are absent'
  })
  await check('Production Markdown worker backpressure and theme', async () => {
    const code = '```swift\nlet value = "東京"\n```\n\nparity-end'
    await load(code, false)
    await until(() => host.value?.querySelector('.pk-code .shiki'), 'Swift highlighting missing')
    const oldStyle = host.value!.querySelector('.pk-code .shiki')!.getAttribute('style')
    dark.value = !dark.value
    await until(() => {
      const pre = host.value?.querySelector('.pk-code .shiki')
      return pre && pre.getAttribute('style') !== oldStyle
    }, 'Theme change did not replace syntax output')
    streaming.value = true
    // Yield Vue but not the worker event queue: overload must stay latest-only.
    for (let i = 0; i < 500; i++) { content.value = `**Burst ${i}**\n\n${code}`; await nextTick() }
    content.value = `**Newest burst**\n\n${code}`; streaming.value = false; await nextTick()
    await until(() => ready()?.textContent?.includes('Newest burst') && ready()?.dataset.markdownFinal === 'true', 'Newest final response was lost under backpressure')
    assert(markdownParser.stats.workers === 1 && markdownParser.stats.queued <= 1, 'Parser worker/queue multiplied under burst load')
    assert(syntaxHighlighter.stats.workers <= 1, 'Syntax worker multiplied per code block')
    assert(host.value!.querySelector('button[aria-label="Copy code"]'), 'Accessible code copy control missing')
    const range = document.createRange(); range.selectNodeContents(host.value!.querySelector('.pk-code pre code')!)
    assert(range.toString() === 'let value = "東京"\n', `Code selection differs from the fenced source, including its trailing newline: ${JSON.stringify(range.toString())}`)
    dark.value = false
    return '500 queued Vue commits keep one parser worker and latest final response; shared syntax worker, theme change, copy control and DOM Range code selection verified (not a physical clipboard/VoiceOver test)'
  })
  await check('Empty Markdown releases its parser lease and safely reopens', async () => {
    content.value = ''; streaming.value = false; await nextTick()
    assert(markdownParser.stats.owners === 0 && markdownParser.stats.workers === 0,
      'The cleared, still-mounted Markdown view retained a parser consumer/worker')
    assert(ready()?.textContent === '', 'Clearing Markdown retained old rich content')
    for (let cycle = 0; cycle < 3; cycle++) {
      await load(`**Reopened ${cycle}**\n\nparity-end`, false)
      assert(host.value!.querySelector('strong')?.textContent === `Reopened ${cycle}`, 'Reopen painted an old revision')
      assert(Number(markdownParser.stats.owners) === 1 && Number(markdownParser.stats.workers) === 1,
        'Reopened Markdown did not acquire exactly one parser consumer')
      content.value = ''; await nextTick()
      assert(markdownParser.stats.owners === 0 && markdownParser.stats.workers === 0, 'Reopen leaked a parser lease')
    }
    return 'Three populate/clear cycles on the same mounted component release the last worker immediately and reopen with fresh content; no GC or idle-timeout change'
  })
}
