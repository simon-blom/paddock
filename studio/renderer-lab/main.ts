import { createApp } from 'vue'
import { reportError } from './report'
import './style.css'

window.addEventListener('error', e => reportError('JavaScript error', e.error ?? e.message))
window.addEventListener('unhandledrejection', e => reportError('Unhandled rejection', e.reason))
const warn = console.warn.bind(console)
console.warn = (...args: unknown[]) => {
  warn(...args)
  reportError('Renderer warning', args.map(value => value instanceof Error ? value.stack ?? value.message : String(value)).join(' '))
}

// Keep diagnostics alive even when a renderer's module fails during initialization.
import('./Lab.vue').then(({ default: Lab }) => {
  const app = createApp(Lab)
  app.config.errorHandler = error => reportError('Vue error', error)
  app.mount('#app')
}).catch(error => reportError('Renderer bootstrap', error))
