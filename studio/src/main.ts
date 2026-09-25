import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'

import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import './styles/base.css'
import './styles/components.css'

import { initializeStudioPersistence } from '@/lib/browser-storage-migration'
import { installPreferenceLifecycle } from '@/lib/ui-preferences'

async function start() {
  const root = document.getElementById('app')!
  try {
    await initializeStudioPersistence()
    installPreferenceLifecycle()
    createApp(App).use(createPinia()).use(router).mount(root)
  } catch {
    // Do not mount default preferences over a database we failed to read.
    const message = document.createElement('p')
    message.textContent = 'Studio could not open its saved data.'
    const retry = document.createElement('button')
    retry.textContent = 'Retry'
    retry.onclick = () => { retry.disabled = true; void start() }
    root.replaceChildren(message, retry)
  }
}
void start()
