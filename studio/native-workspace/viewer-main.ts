import { createApp, h } from 'vue'
import { createPinia } from 'pinia'
import ViewerOnly from './ViewerOnly.vue'
import { installNativeTheme } from './theme'
import { uiPreferences, installPreferenceLifecycle } from '@/lib/ui-preferences'
import PersistenceStatus from '@/components/ui/PersistenceStatus.vue'
import type { Conversation, GraphPart, Message } from '@/types/chat'
import '@/styles/base.css'
import '@/styles/components.css'
import './style.css'
installNativeTheme()
// WKWebView uses a fresh, nonpersistent data store. Load durable viewer
// preferences directly from SQLite; never reconstruct browser storage.
async function start() {
  try {
    await uiPreferences.initialize()
    installPreferenceLifecycle()
    createApp({ render: () => [h(ViewerOnly), h(PersistenceStatus)] }).use(createPinia()).mount('#app')
  } catch {
    const root = document.getElementById('app')!
    const message = document.createElement('p')
    message.textContent = 'The viewer could not open its saved settings.'
    const retry = document.createElement('button')
    retry.textContent = 'Retry'
    retry.onclick = () => { retry.disabled = true; void start() }
    root.replaceChildren(message, retry)
  }
}
void start()
declare global {
  interface Window {
    paddockViewer: {
      update(value: { document?: Conversation; graph?: { id: string; title: string; body: string } | null; graphSource?: GraphPart; graphHistory?: Message[]; conversationId: string; visibleGraph?: boolean }, dark: boolean): Promise<string>
      action(name: string): void
      close(): void
    }
  }
}
