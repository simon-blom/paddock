import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ViewerOnly from './ViewerOnly.vue'
import { installNativeTheme } from './theme'
import type { Conversation, GraphPart } from '@/types/chat'
import '@/styles/base.css'
import '@/styles/components.css'
import './style.css'
installNativeTheme()
createApp(ViewerOnly).use(createPinia()).mount('#app')
declare global {
  interface Window {
    paddockViewer: {
      update(value: { document?: Conversation; graph?: { id: string; title: string; body: string } | null; graphSource?: GraphPart; conversationId: string; visibleGraph?: boolean }, dark: boolean): Promise<string>
      action(name: string): void
      close(): void
    }
  }
}
