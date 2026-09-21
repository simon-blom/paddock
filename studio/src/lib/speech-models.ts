import { friendlyModelName } from './model-caps'
import type { MicMode } from './audio-policy'

export const DICTATION_SETUP = 'Dictation needs a speech model running. Start one and the mic will type what you say straight into the composer.'

/** Presentation shared by SpeechModels.vue and the native microphone menu.
 * Deliberately excludes saved config, paths and credentials. */
export function speechModelRows(rows: readonly {
  port: number; model: string | null; display?: string | null; vendor?: string | null
  running: boolean; busy: boolean
}[], pending: number | null = null) {
  return rows.map(r => ({
    port: r.port, model: r.model, title: r.display ?? friendlyModelName(r.model ?? ''), vendor: r.vendor ?? '',
    running: r.running, busy: r.busy || pending === r.port,
    status: pending === r.port || r.busy ? 'working...' : r.running ? `running · port ${r.port}` : `port ${r.port}`,
    canStart: !r.running && !r.busy && pending === null,
    canStop: r.running && !r.busy && pending === null,
  }))
}

export function microphoneMenu(p: {
  jobs: readonly MicMode[]; mode: MicMode; audioMode: boolean; liveImpossible: boolean
  namedDevices: boolean; devices: number; ears: number; speech: number; docParser: boolean
}) {
  const jobChoice = p.jobs.length > 1 || (p.audioMode && p.liveImpossible)
  const deviceChoice = p.namedDevices && p.devices > 1
  const earChoice = p.mode === 'dictate' && !p.docParser && p.ears > 1
  return { jobChoice, deviceChoice, earChoice,
    offered: p.jobs.length > 0,
    menu: jobChoice || deviceChoice || earChoice || p.speech > 0,
    needsSetup: !p.jobs.length && !p.audioMode && !p.docParser && !p.ears,
    setupMessage: DICTATION_SETUP,
    setupAction: p.speech ? 'Start a new speech model' : 'Start a speech model',
  }
}
