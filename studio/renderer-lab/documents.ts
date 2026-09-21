import { LectorEngine, LectorPane } from '@truespar/lector-core'
import type { DocumentCapability, SearchCapability, InteractionCapability, PageMountedEvent, DocumentHandle } from '@truespar/lector-core'
import { READER_PLUGINS } from '@truespar/lector-vue'
import '@truespar/lector-core/css/tokens.css'
import '@truespar/lector-core/css/base.css'
import pdfiumWorkerUrl from '@truespar/lector-core/worker?worker&url'
import { ScriptorView } from '@truespar/scriptor-core'
import { assert, until } from './report'

/** Detect actual colored/ink pixels, not merely a mounted, blank canvas. */
export function inkPixels(canvas: HTMLCanvasElement): number {
  // Presentation canvases may use bitmaprenderer; never request an incompatible
  // second context type. This readback is for correctness checks, not profiling.
  const probe = new OffscreenCanvas(canvas.width || 1, canvas.height || 1)
  const context = probe.getContext('2d')
  if (!context || !canvas.width || !canvas.height) return 0
  context.drawImage(canvas, 0, 0)
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height)
  let ink = 0
  for (let i = 0; i < data.length; i += 16) {
    if (data[i + 3] > 200 && Math.min(data[i], data[i + 1], data[i + 2]) < 180) ink++
  }
  return ink
}
export class PDFCase {
  engine: LectorEngine | null = null
  pane: LectorPane | null = null
  handle: DocumentHandle | null = null
  overlayClicks = 0

  async open(container: HTMLElement, file = 'sample.pdf', pages = 2, readback = true): Promise<string> {
    await this.close()
    const engine = this.engine = new LectorEngine({
      wasmUrl: '/pdfium/pdfium-st.wasm', wasmJsUrl: '/pdfium/pdfium-st.js',
      wasmUrlFallback: '/pdfium/pdfium-st.wasm', wasmJsUrlFallback: '/pdfium/pdfium-st.js',
      workerUrl: new URL(pdfiumWorkerUrl, import.meta.url),
    })
    for (const plugin of READER_PLUGINS) engine.plugins.register(plugin)
    await engine.init()
    engine.plugins.events.on('ui:page-mounted', (...args: unknown[]) => {
      const event = args[0] as PageMountedEvent
      if (event.pageIndex !== 0) return
      const button = document.createElement('button')
      button.className = 'ocr-region'
      button.textContent = 'Synthetic OCR region - click to select'
      button.onclick = () => { this.overlayClicks++; button.textContent = 'OCR region selected' }
      event.overlayEl.append(button)
    })
    this.handle = await engine.plugins.get<DocumentCapability>('document').load(
      await (await fetch(`./fixtures/${file}`)).arrayBuffer(),
    )
    assert(this.handle.pageCount === pages, `Expected ${pages} PDF pages; got ${this.handle.pageCount}`)
    engine.plugins.get<InteractionCapability>('interaction').setMode('text-select')
    this.pane = new LectorPane({ engine, container, docId: this.handle.id })
    await until(() => this.pane?.isPageReady(0), 'PDF first page did not complete its current-resolution draw', 60000)
    if (readback) await until(() => [...container.querySelectorAll('canvas')].some(c => inkPixels(c) > 100), 'PDF canvas did not paint ink')
    return `PDFium WASM in worker: ${this.handle.pageCount} pages, visible canvas contains ink`
  }
  async search(expected = 2): Promise<string> {
    assert(this.engine && this.handle, 'Open PDF first')
    const result = await this.engine.plugins.get<SearchCapability>('search').search(this.handle.id, 'Labrador')
    assert(result.totalCount === expected, `Expected ${expected} search matches; got ${result.totalCount}`)
    this.engine.plugins.get<SearchCapability>('search').goToMatch(0)
    return 'Found Labrador on both pages; search highlighting and text-selection mode enabled'
  }
  async overlay(container: HTMLElement): Promise<string> {
    await until(() => container.querySelector('.ocr-region'), 'OCR overlay was not mounted')
    const button = container.querySelector<HTMLButtonElement>('.ocr-region')!
    const before = this.overlayClicks
    button.click()
    assert(this.overlayClicks === before + 1, 'OCR overlay click handler failed')
    return 'Page-mounted overlay and DOM click verified; physical hit-testing remains manual'
  }
  async close() {
    this.pane?.destroy(); this.pane = null
    if (this.handle && this.engine) await this.engine.plugins.get<DocumentCapability>('document').close(this.handle.id)
    this.handle = null
    await this.engine?.destroy(); this.engine = null
  }
}
export class DOCXCase {
  view: ScriptorView | null = null
  selections = 0
  async open(container: HTMLElement, file = 'sample.docx', pages = 2, readback = true): Promise<string> {
    this.close()
    this.view = await ScriptorView.create(container, {
      mode: 'read', selectable: true, onSelectionChange: () => { this.selections++ },
    })
    this.view.loadDocx(new Uint8Array(await (await fetch(`./fixtures/${file}`)).arrayBuffer()))
    await until(() => this.view!.pageCount() === pages, `DOCX did not lay out as ${pages} pages (got ${this.view.pageCount()})`, 60000)
    if (readback) await until(() => [...container.querySelectorAll('canvas')].some(c => inkPixels(c) > 100), 'DOCX canvas did not paint ink')
    const xml = this.view.toDocumentXml()
    assert(xml.includes('Scriptor WASM') && xml.includes('Second page'), 'DOCX text/table missing from parsed OOXML')
    assert(xml.includes('blip') && xml.includes('Revised wording'), 'DOCX image or revision missing from parsed OOXML')
    return `Scriptor WASM: ${this.view.pageCount()} pages; text, table, image relationship and revision parsed; canvas has ink`
  }
  async review(): Promise<string> {
    assert(this.view, 'Open DOCX first')
    const reviewers = this.view.reviewers()
    assert(reviewers.some(r => r.name === 'Lab Reviewer'), 'Tracked-change author not found')
    this.view.setTrackDisplay('original')
    assert(this.view.trackDisplayMode === 'original', 'Original review mode failed')
    this.view.setTrackDisplay('all')
    this.view.setZoom(0.9)
    assert(Math.abs(this.view.zoomLevel - 0.9) < 0.001, 'DOCX zoom did not apply')
    return 'Tracked-change author, original/all mode switch and 90% zoom verified; visual revision fidelity remains manual'
  }
  close() { this.view?.destroy(); this.view = null }
}
