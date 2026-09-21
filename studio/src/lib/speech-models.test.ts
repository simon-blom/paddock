import { describe, it, expect } from 'vitest'
import { microphoneMenu, speechModelRows, DICTATION_SETUP } from './speech-models'
import nativeView from '../../../apps/macos/Sources/PaddockUI/StudioMicrophoneView.swift?raw'
import nativeWorkspace from '../../../apps/macos/Sources/PaddockUI/WorkspaceView.swift?raw'
import nativeActions from '../../../apps/macos/Sources/PaddockUI/DesktopActions.swift?raw'
import webView from '../components/chat/SpeechModels.vue?raw'
import webComposer from '../components/chat/Composer.vue?raw'
import nativeAudio from '../../native-workspace/audio.ts?raw'

const base = { jobs: [] as ('live'|'record'|'dictate')[], mode: 'dictate' as const, audioMode: false,
  liveImpossible: true, namedDevices: false, devices: 0, ears: 0, speech: 0, docParser: false }
const saved = { port: 11542, model: 'whisper', display: 'Whisper Large V3', vendor: 'OpenAI', running: false, busy: false }
describe('web/native speech setup parity', () => {
  it('has one empty-state explanation and setup action, not a generic settings form', () => {
    expect(microphoneMenu(base)).toMatchObject({ needsSetup: true, offered: false, menu: false,
      setupMessage: DICTATION_SETUP, setupAction: 'Start a speech model', deviceChoice: false, earChoice: false, jobChoice: false })
    expect(nativeView).not.toContain('Model catalog…')
    expect(nativeView).toContain('workspace?.request(.startSpeechModel)')
    expect(nativeView).not.toContain('workspace?.request(.startModel)')
    expect(nativeActions).toMatch(/case \.startSpeechModel:\s*navigation\.showModelLibrary\(purpose: \.speech\)/)
    expect(nativeWorkspace).toContain('purpose: chosen.purpose')
    expect(nativeWorkspace).toContain('navigation.showModelLibrary(purpose: chosen.purpose)')
    expect(webComposer).toContain('{{ DICTATION_SETUP }}')
  })
  it('keeps configured but stopped models available with explicit Start and Stop buttons', () => {
    expect(microphoneMenu({ ...base, speech: 1 })).toMatchObject({ needsSetup: true, setupAction: 'Start a new speech model' })
    expect(speechModelRows([saved])[0]).toMatchObject({ title: 'Whisper Large V3', status: 'port 11542', canStart: true, canStop: false })
    expect(speechModelRows([{ ...saved, running: true }])[0]).toMatchObject({ status: 'running · port 11542', canStart: false, canStop: true })
    expect(nativeView).toContain('Button("Start")')
    expect(nativeView).toContain('Button("Stop")')
  })
  it('shows working and blocks both actions while a start is in flight, including other rows', () => {
    const rows = [saved, { ...saved, port: 11543, running: true }]
    expect(speechModelRows(rows, saved.port)).toMatchObject([
      { busy: true, status: 'working...', canStart: false, canStop: false },
      { canStart: false, canStop: false },
    ])
    expect(speechModelRows([{ ...saved, busy: true }])[0]).toMatchObject({ status: 'working...', canStart: false, canStop: false })
  })
  it('keeps lifecycle actions in the running mic menu without inventing one-choice selectors', () => {
    expect(microphoneMenu({ ...base, jobs: ['dictate'], ears: 1, speech: 1 })).toMatchObject({
      needsSetup: false, offered: true, menu: true, jobChoice: false, deviceChoice: false, earChoice: false })
    expect(microphoneMenu({ ...base, jobs: ['dictate'], ears: 2 })).toMatchObject({ earChoice: true, menu: true })
    expect(microphoneMenu({ ...base, namedDevices: true, devices: 2 })).toMatchObject({ deviceChoice: true })
    expect(microphoneMenu({ ...base, namedDevices: false, devices: 2 }).deviceChoice).toBe(false)
  })
  it('offers file-only cloud recording, not the no-speech fallback or a second mic', () => {
    expect(microphoneMenu({ ...base, jobs: ['record'], mode: 'record', audioMode: true })).toMatchObject({
      needsSetup: false, offered: true, jobChoice: true, menu: true })
    expect(microphoneMenu({ ...base, docParser: true }).needsSetup).toBe(false)
  })
  it('both presenters use the same redacted row projection and conditional menu policy', () => {
    expect(webView).toContain('speechModelRows(fleet.speechEndpoints, pending.value)')
    expect(nativeAudio).toContain('speechModelRows(fleet.speechEndpoints)')
    expect(nativeAudio).toContain('microphoneMenu(')
    expect(webComposer).toContain('microphoneMenu(')
    const row = speechModelRows([{ ...saved, weights: 'private-path', api_key: 'private-key' } as typeof saved])[0]
    expect(JSON.stringify(row)).not.toContain('private-')
  })
})
