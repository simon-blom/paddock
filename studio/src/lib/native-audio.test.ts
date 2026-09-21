import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import { ref, nextTick } from 'vue'
const f = vi.hoisted(() => ({ models: {} as any, chat: {} as any, settings: {} as any, mic: {} as any, recorder: {} as any, live: {} as any, fleet: {} as any, devices: {} as any }))
vi.mock('@/stores/models', () => ({ useModelsStore: () => f.models }))
vi.mock('@/stores/chat', () => ({ useChatStore: () => f.chat }))
vi.mock('@/stores/settings', () => ({ useSettingsStore: () => f.settings }))
vi.mock('@/stores/fleet', () => ({ useFleetStore: () => f.fleet }))
vi.mock('@/composables/useMicTranscribe', () => ({ DICTATION_IDLE_MS: 5000, useMicTranscribe: () => f.mic }))
vi.mock('@/composables/useRecorder', () => ({ useRecorder: () => f.recorder }))
vi.mock('@/composables/useLiveTurn', () => ({ useLiveTurn: () => f.live }))
vi.mock('@/composables/useAudioDevices', () => ({ useAudioDevices: () => f.devices }))
vi.mock('../../native-workspace/preferences', () => ({ savePreferences: vi.fn() }))
vi.mock('@/lib/languages', () => ({ askedLanguage: (s: string) => s === 'auto' ? undefined : s || 'sv', localeLanguage: () => 'sv', languageOptions: () => [{ value: 'auto', label: 'Auto' }, { value: 'sv', label: 'Swedish' }] }))
import { createNativeAudio } from '../../native-workspace/audio'
function setup() {
  const hooks = { begin: vi.fn(async () => {}), record: vi.fn(async () => ({ accepted: true })), idle: vi.fn() }
  return { audio: createNativeAudio(hooks), hooks }
}
beforeEach(() => {
  vi.useFakeTimers()
  f.devices = { devices: ref([]), lost: ref(null), refresh: vi.fn(), named: () => false, reveal: vi.fn(async () => true) }
  f.fleet = { speechEndpoints: [], error: null, refresh: vi.fn(async () => {}) }
  f.settings = { micDeviceId: '', micDeviceLabel: '', dictateWith: 'speech' }
  f.models = { currentId: 'text', caps: {}, models: [{ id: 'text', port: 1, status: 'ok' }, { id: 'speech', port: 2, status: 'ok' }, { id: 'gen', port: 3, status: 'ok' }, { id: 'cloud', status: 'ok' }],
    canChat: (id: string) => ['text', 'gen'].includes(id), canTranscribe: (id: string) => id !== 'text', transcribeStreams: (id: string) => ['speech', 'gen'].includes(id),
    portFor: (id: string) => f.models.models.find((m: any) => m.id === id)?.port, canTimeWords: (id: string) => id === 'speech' }
  f.chat = { active: { id: 'conversation', model: 'text', audioLanguage: 'sv' }, persist: vi.fn() }
  f.mic = { lanes: ref([]), listening: ref(false), idle: ref(false), error: ref(null), levels: ref([]),
    start: vi.fn(async () => { f.mic.listening.value = true }), stop: vi.fn(async () => ({ lanes: f.mic.lanes.value })), cancel: vi.fn() }
  f.recorder = { levels: ref([]), elapsed: ref(0), remaining: ref(3600), capped: ref(false), recording: ref(false), arming: ref(false), error: ref(null),
    start: vi.fn(async () => true), stop: vi.fn(async () => new File(['clip'], 'recording.wav')), cancel: vi.fn() }
  f.live = { begin: vi.fn(() => true), apply: vi.fn(), finish: vi.fn(async () => {}), abandon: vi.fn() }
})
afterEach(() => { vi.clearAllTimers(); vi.useRealTimers() })
describe('native adapter reuses the web speech engine', () => {
  it('device enumeration requires an explicit action and never transcribes', async () => {
    const { audio, hooks } = setup()
    await audio.refresh()
    expect(f.devices.reveal).not.toHaveBeenCalled()
    await audio.revealDevices()
    expect(f.devices.reveal).toHaveBeenCalledOnce()
    expect(f.mic.start).not.toHaveBeenCalled()
    expect(f.recorder.start).not.toHaveBeenCalled()
    expect(hooks.record).not.toHaveBeenCalled()
    f.devices.reveal.mockResolvedValue(false)
    await expect(audio.revealDevices()).rejects.toThrow('Microphone access')
    await audio.start()
    await expect(audio.revealDevices()).rejects.toThrow('Stop the microphone')
    audio.cancel()
  })
  it('bounds a suspended capture start, releases engines and allows a retry', async () => {
    let release!: () => void
    f.mic.start.mockImplementationOnce(() => new Promise<void>(resolve => { release = resolve }))
    const { audio } = setup()
    const starting = audio.start()
    expect(audio.state().phase).toBe('starting')
    await vi.advanceTimersByTimeAsync(15_000)
    await starting
    expect(audio.state()).toMatchObject({ phase: 'idle', error: expect.stringContaining('Audio settings') })
    expect(f.mic.cancel).toHaveBeenCalledOnce()
    release(); await nextTick()
    expect(audio.state().phase).toBe('idle')
    await audio.start()
    expect(audio.state().phase).toBe('listening')
    audio.cancel()
  })
  it('refreshes configured speech inventory without recording or submitting a turn', async () => {
    f.models.models = f.models.models.filter((m: any) => m.id === 'text')
    f.fleet.speechEndpoints = [{ port: 2, model: 'speech', display: 'Speech', running: false, busy: false }]
    const { audio, hooks } = setup()
    await audio.refresh()
    expect(f.fleet.refresh).toHaveBeenCalledOnce()
    expect(audio.state().menu).toMatchObject({ needsSetup: true, setupAction: 'Start a new speech model' })
    expect(audio.state().speechModels[0]).toMatchObject({ port: 2, canStart: true, canStop: false })
    expect(f.mic.start).not.toHaveBeenCalled(); expect(f.recorder.start).not.toHaveBeenCalled()
    expect(hooks.record).not.toHaveBeenCalled()
  })
  it('dictates finalized items exactly once without sending or retaining a recording', async () => {
    const { audio, hooks } = setup()
    await audio.start()
    expect(f.mic.start.mock.calls[0][0]).toMatchObject({ ports: [2], language: 'sv', record: false, detail: [false], idleMs: 5000 })
    f.mic.lanes.value = [{ items: [], open: 'Provisional' }]; await nextTick()
    expect(audio.state().dictation).toEqual([])
    f.mic.lanes.value[0].items.push({ text: 'Hej världen.' }); await nextTick()
    expect(audio.state().dictation).toEqual([{ index: 0, text: 'Hej världen.' }])
    audio.acknowledge({ session: 'stale', index: 0 }); expect(audio.state().dictation).toHaveLength(1)
    audio.acknowledge({ session: audio.state().session, index: 0 }); expect(audio.state().dictation).toEqual([])
    await audio.stop('Existing draft'); expect(hooks.record).not.toHaveBeenCalled(); expect(f.live.begin).not.toHaveBeenCalled()
    audio.cancel()
  })
  it('opens both Whisper and generative live lanes with per-lane timestamp capabilities', async () => {
    f.chat.active.compareModels = ['speech', 'gen']
    const { audio, hooks } = setup(); await audio.start()
    expect(hooks.begin).toHaveBeenCalledOnce()
    expect(f.mic.start.mock.calls[0][0]).toMatchObject({ ports: [2, 3], detail: [true, false], record: true })
    await audio.stop(''); expect(f.live.finish).toHaveBeenCalledOnce(); expect(hooks.record).not.toHaveBeenCalled()
    audio.cancel()
  })
  it('records for local/cloud Compare without dropping the cloud lane or opening realtime', async () => {
    f.chat.active.compareModels = ['speech', 'cloud']
    const { audio, hooks } = setup(); expect(audio.state().liveBlocked).toBe(true)
    await audio.start(); await audio.stop('')
    expect(f.mic.start).not.toHaveBeenCalled(); expect(hooks.record).toHaveBeenCalledOnce()
    audio.cancel()
  })
  it('cancels permission-in-flight without leaving capture active', async () => {
    let opened!: () => void
    f.mic.start.mockImplementation(() => new Promise<void>(r => { opened = r }))
    const { audio } = setup(); const start = audio.start()
    audio.cancel(); opened(); await start
    expect(audio.state().phase).toBe('idle'); expect(f.mic.cancel).toHaveBeenCalled()
  })
  it('preserves explicit auto language and rejects unsupported settings', async () => {
    const { audio } = setup(); await audio.configure({ language: 'auto' }); await audio.start()
    expect(f.mic.start.mock.calls[0][0].language).toBeUndefined()
    await expect(audio.configure({ device: 'missing' })).rejects.toThrow('Stop the microphone')
    audio.cancel(); await expect(audio.configure({ device: 'missing' })).rejects.toThrow('not connected')
    await expect(audio.configure({ language: 'invented' })).rejects.toThrow('Unknown speech language')
  })
  it('stops capped recordings even when native controls are not mounted, retaining the clip', async () => {
    f.chat.active.model = 'cloud'
    const { audio, hooks } = setup(); await audio.start()
    f.recorder.capped.value = true; await nextTick(); await nextTick()
    expect(f.recorder.stop).toHaveBeenCalledOnce()
    expect(hooks.record).not.toHaveBeenCalled()
    expect(audio.state().retryAvailable).toBe(true)
    await audio.stop(''); expect(hooks.record).toHaveBeenCalledOnce()
    expect(audio.state().retryAvailable).toBe(false)
    audio.cancel()
  })
  it('retains failed recording submissions for explicit retry without recapturing', async () => {
    f.chat.active.model = 'cloud'
    const { audio, hooks } = setup(); hooks.record.mockRejectedValueOnce(new Error('Store unavailable'))
    await audio.start(); await audio.stop('')
    expect(audio.state()).toMatchObject({ retryAvailable: true, error: 'Store unavailable' })
    await audio.stop('')
    expect(f.recorder.stop).toHaveBeenCalledOnce(); expect(hooks.record).toHaveBeenCalledTimes(2)
    audio.cancel()
  })
})
