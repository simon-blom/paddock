import { createApp, h } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import { createMemoryHistory, createRouter, RouterView } from 'vue-router'
import { useModelsStore } from '@/stores/models'
import { useFleetStore } from '@/stores/fleet'
import { useRegistryStore } from '@/stores/registry'
import { usePushStore } from '@/stores/push'
import EmbeddedViewers from './EmbeddedViewers.vue'
import { NATIVE_CONTENT } from '@/lib/native-content'
import { createController } from './controller'
import { restorePreferences } from './preferences'
import { installNativeTheme } from './theme'
import '@/styles/base.css'
import '@/styles/components.css'
import './style.css'

installNativeTheme()
await restorePreferences()
const router = createRouter({ history: createMemoryHistory(), routes: [
  { path: '/', redirect: '/studio' },
  { path: '/studio', name: 'home', component: EmbeddedViewers },
  { path: '/studio/chat/new', name: 'chat-new', component: EmbeddedViewers },
  { path: '/studio/chat/:id', name: 'chat', component: EmbeddedViewers },
  ...['servers', 'server-edit', 'server-detail', 'server-new', 'gpus', 'cloud', 'connectors', 'embeddings', 'prompts', 'settings'].map(name => ({ path: `/native/${name}/:port?`, name, component: { render: () => null }, beforeEnter: () => { throw new Error('Use the native Manager to configure models and connections') } })),
] })
const pinia = createPinia()
setActivePinia(pinia)
const controller = createController(router)
createApp({ render: () => h(RouterView) }).use(pinia).use(router).provide(NATIVE_CONTENT, controller.host).mount('#app')
await router.isReady()
await controller.initialized
window.paddockWorkspace = controller
controller.publish()
void useRegistryStore().refresh()
void useFleetStore().refresh()
usePushStore().connect()
// The native Manager can change the fleet while this content is retained.
// Push is the primary path; reconcile cloud choices when the window returns.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') void useModelsStore().refresh()
})
declare global { interface Window { paddockWorkspace: ReturnType<typeof createController> } }
