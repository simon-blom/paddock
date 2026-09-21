import { describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { createSSRApp, h } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { TooltipProvider } from 'reka-ui'
import { useReadinessStore, type Readiness } from './readiness'
import BackendStatus from '@/components/layout/BackendStatus.vue'
import ReadinessNotice from '@/components/manage/ReadinessNotice.vue'
import headerSource from '@/components/layout/AppHeader.vue?raw'

const fixture = (state: Readiness['state'], backend = 'metal'): Readiness => ({
  backend, state, card: 'Apple M5 Max', os: 'macos', cuda_needed: '', supported: [],
})
function setup(info: Readiness | null) {
  const pinia = createPinia()
  setActivePinia(pinia)
  const readiness = useReadinessStore()
  readiness.info = info
  const app = createSSRApp({ render: () => h(TooltipProvider, {}, { default: () => [h(BackendStatus), h(ReadinessNotice)] }) })
  app.use(pinia)
  app.component('RouterLink', { render: () => h('a') })
  return { app, readiness }
}

describe('compact Metal readiness', () => {
  it('renders one neutral header indicator, without a preview card or first-run split', async () => {
    const { app, readiness } = setup(fixture('untested'))
    expect(readiness.blocked).toBe(false)
    expect(readiness.hasMetrics).toBe(true)
    expect(readiness.notice).toBeNull()
    expect(readiness.headerStatus).toMatchObject({ label: 'Metal' })
    expect(readiness.headerStatus?.detail).toContain('Apple M5 Max · Metal preview')
    const html = await renderToString(app)
    expect(html).toContain('backend-status')
    expect(html).toContain('tabindex="0"')
    expect(html).toContain('Metal backend information')
    expect(html).not.toContain('class="rn')
    expect(html).not.toContain('can run Metal previews')
    expect(headerSource).toContain('<BackendStatus />')
  })
  it('keeps unsupported hardware actionable instead of hiding the blocker in a tooltip', async () => {
    const { app, readiness } = setup(fixture('no-card'))
    expect(readiness.blocked).toBe(true)
    expect(readiness.headerStatus).toBeNull()
    expect(readiness.notice?.state).toBe('no-card')
    const html = await renderToString(app)
    expect(html).toContain('rn--no-card')
    expect(html).toContain('requires an Apple Silicon GPU')
    expect(html).not.toContain('backend-status')
  })
  it('does not change CUDA notices or invent a Metal status before discovery', () => {
    const { readiness } = setup(null)
    expect(readiness.headerStatus).toBeNull()
    for (const state of ['untested', 'no-card', 'driver-too-old'] as const) {
      readiness.info = fixture(state, 'cuda')
      expect(readiness.notice?.state).toBe(state)
      expect(readiness.headerStatus).toBeNull()
    }
    readiness.info = fixture('ready')
    expect(readiness.headerStatus?.label).toBe('Metal')
    expect(readiness.headerStatus?.detail).not.toContain('preview')
    expect(readiness.notice).toBeNull()
  })
})
