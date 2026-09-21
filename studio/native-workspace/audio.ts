import { computed, ref, shallowRef, watch } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useModelsStore } from '@/stores/models'
import { useSettingsStore } from '@/stores/settings'
import { useFleetStore } from '@/stores/fleet'
import { microphoneMenu, speechModelRows } from '@/lib/speech-models'
import { useAudioDevices } from '@/composables/useAudioDevices'
import { useMicLevels } from '@/composables/useMicLevels'
import { useMicTranscribe, DICTATION_IDLE_MS, type MicResult } from '@/composables/useMicTranscribe'
import { useRecorder } from '@/composables/useRecorder'
import { useLiveTurn } from '@/composables/useLiveTurn'
import { audioPolicy, type MicMode } from '@/lib/audio-policy'
import { askedLanguage, languageOptions, localeLanguage } from '@/lib/languages'
import { savePreferences } from './preferences'
import { string } from './protocol'
import { NATIVE_RECORDING_TYPES } from '@/lib/audio-recording'

/** Native controls, shared media engine. No PCM, recording bytes, secrets or
 * arbitrary destinations cross the Swift presentation bridge. */
export function createNativeAudio(hooks: {
  begin(): Promise<void>
  record(file: File, text: string): Promise<unknown>
  idle(): void
}) {
  const chat = useChatStore(), models = useModelsStore(), settings = useSettingsStore()
  const fleet = useFleetStore()
  const mic = useMicTranscribe(), recorder = useRecorder(), live = useLiveTurn(), devices = useAudioDevices()
  const meter = useMicLevels()
  meter.setFrameDriven(false)
  const mode = ref<MicMode>('live'), phase = ref('idle'), failure = ref(''), session = ref('')
  const dictation = ref<{ index: number; text: string }[]>([])
  const pendingRecording = shallowRef<File | null>(null)
  let epoch = 0, drained = 0, stopping: Promise<unknown> | null = null
  let activeMode: MicMode = 'dictate', activeConversation = ''
  const stopIdle = ref(false), levels = ref<readonly number[]>([]), elapsed = ref(0)
  let meterTimer: ReturnType<typeof setInterval> | undefined
  const ids = computed(() => chat.active?.compareModels?.length ? chat.active.compareModels : [chat.active?.model ?? models.currentId].filter(Boolean))
  const ears = computed(() => models.models.filter(m => m.port && m.status === 'ok' && models.canTranscribe(m.id) && models.transcribeStreams(m.id)))
  const ear = computed(() => ears.value.find(m => m.id === settings.dictateWith) ?? ears.value[0])
  const policy = computed(() => audioPolicy(ids.value.map(id => ({ chat: models.canChat(id), audio: models.canTranscribe(id), live: models.transcribeStreams(id) })), ears.value.length, !!models.caps[ids.value[0]]?.docParser))
  const chosen = computed(() => policy.value.jobs.includes(mode.value) ? mode.value : policy.value.jobs[0] ?? 'dictate')
  const busy = computed(() => phase.value !== 'idle')
  watch(busy, active => {
    clearInterval(meterTimer); meterTimer = undefined; levels.value = []; elapsed.value = 0
    if (active) {
      const start = performance.now()
      meterTimer = setInterval(() => {
        meter.sample()
        levels.value = [...(activeMode === 'record' ? recorder.levels.value : mic.levels.value)]
        elapsed.value = (performance.now() - start) / 1000
      }, 100)
    }
  }, { flush: 'sync' })
  const limit = computed(() => {
    const values = ids.value.map(id => models.caps[id]?.transcriptionMaxClipS).filter((v): v is number => typeof v === 'number' && v > 0)
    return values.length ? Math.min(...values) : undefined
  })
  // Resource ceilings also apply while Swift is showing Manager or another
  // window. Stop capture here; an unsent capped recording stays retryable.
  watch([recorder.capped, stopIdle, elapsed], ([capped, quiet, seconds]) => {
    if (!busy.value || stopping) return
    if (activeMode === 'record' && capped) void stop('', false)
    else if (quiet || seconds >= 3600) void stop('')
  })
  function drain() {
    if (activeMode !== 'dictate') return
    const items = mic.lanes.value[0]?.items ?? []
    for (; drained < items.length; drained++) {
      const text = items[drained].text
      if (text) dictation.value.push({ index: drained, text })
    }
    // A disconnected native consumer must not accumulate an unbounded draft.
    if (dictation.value.reduce((n, i) => n + i.text.length, 0) > 48 * 1024) {
      failure.value = 'Dictation paused because the editor is not accepting text. Finish editing before continuing.'
      stopIdle.value = true
    }
  }
  watch(() => mic.lanes.value, lanes => {
    if (!busy.value) return
    if (activeMode === 'live') live.apply(lanes)
    else drain()
  }, { deep: true })
  watch(mic.idle, quiet => {
    if (quiet && activeMode === 'dictate' && mic.lanes.value[0]?.items.length) stopIdle.value = true
  })
  function state() {
    const selectedDevice = settings.micDeviceId
    const inputs = devices.devices.value.filter(d => d.label).map(d => ({ ...d, available: true }))
    if (selectedDevice && !inputs.some(d => d.id === selectedDevice)) inputs.push({ id: selectedDevice, label: settings.micDeviceLabel || 'Chosen microphone', available: false })
    const speechModels = speechModelRows(fleet.speechEndpoints)
    const fileOnly = ids.value.filter(id => !models.transcribeStreams(id)).map(id => models.models.find(m => m.id === id)?.display ?? id)
    const menu = microphoneMenu({ jobs: policy.value.jobs, mode: chosen.value, audioMode: policy.value.audioMode,
      liveImpossible: policy.value.liveBlocked || !ids.value.length, namedDevices: devices.named(),
      devices: inputs.length, ears: ears.value.length, speech: speechModels.length, docParser: !!models.caps[ids.value[0]]?.docParser })
    return {
      menu, speechModels, speechError: fleet.error ?? '',
      mode: busy.value ? activeMode : chosen.value, phase: phase.value,
      retryAvailable: !!pendingRecording.value && !busy.value,
      jobs: policy.value.jobs, audioMode: policy.value.audioMode, audioOk: policy.value.audioOk,
      liveBlocked: policy.value.liveBlocked,
      liveReason: policy.value.liveBlocked ? fileOnly.length
        ? `${fileOnly.join(' and ')} ${fileOnly.length > 1 ? 'hear' : 'hears'} a finished file, not a live stream`
        : 'no model is running to stream to' : '',
      transcribers: ears.value.map(m => ({ id: m.id, label: m.display ?? m.id })), transcriber: ear.value?.id ?? '',
      devices: inputs, devicesNamed: devices.named(), device: selectedDevice, language: chat.active?.audioLanguage || localeLanguage() || 'auto', languages: languageOptions(),
      session: session.value, dictation: dictation.value, provisional: activeMode === 'dictate' ? (mic.lanes.value[0]?.open ?? '').slice(-4096) : '',
      levels: levels.value.slice(0, 24),
      elapsed: activeMode === 'record' ? recorder.elapsed.value : elapsed.value, remaining: recorder.remaining.value, limit: limit.value ?? 3600,
      arming: phase.value === 'starting' || recorder.arming.value, idle: mic.idle.value,
      shouldStop: !!pendingRecording.value && recorder.capped.value && !busy.value,
      error: failure.value || mic.error.value || recorder.error.value || mic.lanes.value.find(l => l.error)?.error || '',
      deviceNote: devices.lost.value ? `${devices.lost.value} is not connected. Recording on the system default instead.` : '',
    }
  }
  async function configure(p: Record<string, unknown>) {
    if (busy.value) throw new Error('Stop the microphone before changing its settings')
    if (p.mode !== undefined) {
      if (!policy.value.jobs.includes(p.mode as MicMode)) throw new Error('This microphone mode is unavailable for the selected models')
      mode.value = p.mode as MicMode
    }
    if (p.transcriber !== undefined) {
      const id = string(p.transcriber, 1024)
      if (!ears.value.some(m => m.id === id)) throw new Error('This dictation model is not running')
      settings.dictateWith = id
    }
    if (p.device !== undefined) {
      const id = string(p.device, 1024), device = devices.devices.value.find(d => d.id === id)
      if (id && !device && id !== settings.micDeviceId) throw new Error('This microphone is not connected')
      settings.micDeviceId = id; settings.micDeviceLabel = device?.label ?? (id ? settings.micDeviceLabel : '')
    }
    if (p.language !== undefined) {
      const value = string(p.language, 32)
      if (!languageOptions().some(l => l.value === value)) throw new Error('Unknown speech language')
      if (chat.active) { chat.active.audioLanguage = value; chat.persist(chat.active) }
    }
    await devices.refresh(); await savePreferences()
  }
  async function finish(out: MicResult, ticket: number) {
    if (ticket !== epoch) return
    if (activeMode === 'live') await live.finish(out)
    else drain()
    if (ticket === epoch) phase.value = 'idle'
  }
  async function start() {
    hooks.idle()
    if (busy.value || dictation.value.length || pendingRecording.value) throw new Error('Finish dictation, or retry/discard the pending recording before starting another')
    if (!policy.value.jobs.length) throw new Error('Start a speech model in Manager to use dictation')
    if (ids.value.some(id => !models.models.some(m => m.id === id && m.status === 'ok'))) throw new Error('Select a reachable model first')
    activeMode = chosen.value; failure.value = ''; stopIdle.value = false; drained = 0
    session.value = crypto.randomUUID(); const ticket = ++epoch
    activeConversation = chat.active?.id ?? ''; phase.value = 'starting'
    let startupTimer: ReturnType<typeof setTimeout> | undefined
    try {
      const capture = async () => {
      if (activeMode === 'record') {
        if (!await recorder.start(limit.value, NATIVE_RECORDING_TYPES)) throw new Error(recorder.error.value || 'The microphone did not start')
      } else {
        const armed = activeMode === 'live' ? ids.value.map(model => ({ model, port: models.portFor(model)! })) : [{ model: ear.value!.id, port: ear.value!.port! }]
        if (activeMode === 'live') { await hooks.begin(); activeConversation = chat.active?.id ?? ''; if (ticket !== epoch) return; if (!live.begin(armed, askedLanguage(chat.active?.audioLanguage))) throw new Error('The transcription turn could not be opened') }
        await mic.start({ ports: armed.map(l => l.port), language: askedLanguage(chat.active?.audioLanguage),
          record: activeMode === 'live', detail: armed.map(l => activeMode === 'live' && models.canTimeWords(l.model)),
          recordingTypes: NATIVE_RECORDING_TYPES,
          idleMs: activeMode === 'dictate' ? DICTATION_IDLE_MS : undefined,
          onDied: out => { void finish(out, ticket) },
        })
        if (!mic.listening.value) throw new Error(mic.error.value || 'The microphone did not start')
      }
      }
      await Promise.race([capture(), new Promise<never>((_, reject) => {
        startupTimer = setTimeout(() => reject(new Error('The microphone did not start. Open Audio settings to check the input device and microphone access, then try again.')), 15_000)
      })])
      if (ticket !== epoch) return // The shared engines release stale captures themselves.
      phase.value = 'listening'
    } catch (e) {
      if (ticket === epoch) { failure.value = e instanceof Error ? e.message : String(e); recorder.cancel(); mic.cancel(); live.abandon(); phase.value = 'idle' }
    } finally { clearTimeout(startupTimer) }
  }
  function stop(text: string, submit = true): Promise<unknown> {
    if (stopping) return stopping
    if (!busy.value && !pendingRecording.value) return Promise.resolve({})
    const ticket = epoch
    stopping = (async () => {
      stopIdle.value = false
      failure.value = ''
      if (phase.value === 'starting') { cancel(); return {} }
      phase.value = 'finishing'
      try {
        if (activeMode === 'record') {
          const clip = pendingRecording.value ?? await recorder.stop()
          if (ticket !== epoch) return {}
          if (!clip) throw new Error(recorder.error.value || 'Nothing was recorded')
          pendingRecording.value = clip
          if (chat.active?.id !== activeConversation) throw new Error('The conversation changed. Recording was not sent.')
          // Release audio admission before the ordinary, durable send path.
          phase.value = 'idle'
          if (!submit) return {}
          const result = await hooks.record(clip, text)
          pendingRecording.value = null
          return result
        }
        await finish(await mic.stop(), ticket)
        return {}
      } catch (e) { failure.value = e instanceof Error ? e.message : String(e); return {} }
      finally { if (ticket === epoch) phase.value = 'idle'; stopping = null }
    })()
    return stopping
  }
  function cancel() { ++epoch; recorder.cancel(); mic.cancel(); live.abandon(); pendingRecording.value = null; phase.value = 'idle'; stopIdle.value = false }
  function acknowledge(p: Record<string, unknown>) {
    if (p.session !== session.value || !Number.isSafeInteger(p.index)) return
    dictation.value = dictation.value.filter(i => i.index > Number(p.index))
  }
  function recordingAccepted() { pendingRecording.value = null }
  async function refresh() { await Promise.all([devices.refresh(), fleet.refresh()]) }
  async function revealDevices() {
    if (busy.value) throw new Error('Stop the microphone before changing its settings')
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      const allowed = await Promise.race([devices.reveal(), new Promise<boolean>(resolve => {
        timer = setTimeout(() => resolve(false), 15_000)
      })])
      if (!allowed) throw new Error('Microphone access is unavailable. Check Audio settings > Microphone access in macOS, then try again.')
    } finally { clearTimeout(timer) }
  }
  return { state, busy, start, stop, cancel, configure, acknowledge, recordingAccepted, refresh, revealDevices }
}
