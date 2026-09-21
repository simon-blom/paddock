import { createApp } from 'vue'
import './style.css'

// The host awaits this explicit module boundary; didFinish alone doesn't mean
// async renderer imports have installed their entry points. No content->native
// message handler or network client is installed in this realm.
declare global { interface Window { paddockTranscriptDidMount?: () => void } }
void import('./Transcript.vue').then(({ default: Transcript }) => {
  createApp(Transcript).mount('#app')
  window.paddockTranscriptDidMount?.()
})
