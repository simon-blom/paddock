import { computed, nextTick, ref, watch, shallowReactive } from 'vue'
import type { Router } from 'vue-router'
import { takesTurns, useModelsStore } from '@/stores/models'
import { useChatStore } from '@/stores/chat'
import { useGraphsStore } from '@/stores/graphs'
import { useConnectorsStore } from '@/stores/connectors'
import { builtinKey, connectorKey, serverKey, useMcpToolsStore } from '@/stores/mcpTools'
import { useSettingsStore } from '@/stores/settings'
import { useArtifactsStore } from '@/stores/artifacts'
import { useInjectedPrompt } from '@/composables/useInjectedPrompt'
import { useChatStream, stopAllStreams, toolSelection, conversationBusy } from '@/composables/useChatStream'
import { historyPage, historyQuery, type HistoryQuery } from './history'
import { selectStudioModel } from '@/lib/select-model'
import { studioModelHeader } from '@/lib/studio-model-header'
import { isDocxPart, isPdfPart, docContext } from '@/lib/docrun'
import { selectedAttachment } from './attachment-selection'
import { DEFAULT_PARAMS, type AudioPart, type ContentPart, type Conversation, type SamplingParams, type ToolSelection } from '@/types/chat'
import type { NativeContentHost } from '@/lib/native-content'
import { describeAttachment } from './attachments'
import { VERSION, attachmentChoices, identifier, object, parseCommand, storedAttachment, string } from './protocol'
import { savePreferences } from './preferences'
import { promptCommand, presetText } from './prompt-library'
import { preferencePresentation, saveStudioPreferences } from './studio-preferences'
import { composerPresentation } from './composer'
import { contextTokens } from '@/lib/tokens'
import { validateSamplingPatch, samplerDefaults } from '@/lib/composer-policy'
import { nativeTranscript } from './native-transcript'
import { documentPreview, sentDocument } from './documents'
import { desktopActivity } from './activity'
import { resolveMessageAction } from './message-actions'
import { changeToolPicker, pickerGroupState, pickerToolChecked, toolTermHits, type PickerAction } from '@/lib/tool-picker'
import { createNativeAudio } from './audio'
import { readAudioPart } from '@/lib/attachments'
import { askedLanguage } from '@/lib/languages'
import { audioBadge, exportTranscript } from './speech'
import { pendingApproval } from './message-extras'
import { approvalsApi } from '@/lib/api'
import { presentationFrames } from './presentation-frames'

/** A presentation adapter, not another chat engine. The shared orchestrator
 * owns trees, compare, tools/approvals, compaction, streaming and persistence.
 * Every view is native except the explicitly embedded document/graph engines.
 * No raw terminal objects, attachment bytes or credentials cross this port. */
export function createController(router: Router) {
  const chat = useChatStore(), models = useModelsStore(), graphs = useGraphsStore()
  const connectors = useConnectorsStore(), tools = useMcpToolsStore(), settings = useSettingsStore()
  const artifacts = useArtifactsStore()
  const injected = useInjectedPrompt(computed(() => chat.active))
  const stream = useChatStream(), preview = ref<Conversation | null>(null)
  const audioPreview = ref<AudioPart | null>(null)
  const pending = ref(false), error = ref(''), viewport = ref({ left: 0, width: 0 })
  const staged = shallowReactive(new Map<string, ContentPart>())
  const recordedAttachment = ref<ContentPart | null>(null)
  const contextUsed = ref(0)
  const completedTurn = ref<{ id: string; conversationId: string; state: string } | null>(null)
  const libraryQuery = ref<HistoryQuery>({ search: '', sort: 'newest', page: 0 })
  const libraryRows = computed(() => historyPage(chat.conversations, libraryQuery.value))
  const recentRows = computed(() => historyPage(chat.conversations, { search: '', sort: 'newest', page: 0 }).rows.slice(0, 20))
  let documentActions: { info(): void; download(): void } | null = null
  const transcript = computed(() => nativeTranscript(chat.active))
  const graphVisible = ref(false)
  const graphArtifact = ref<{ id: string; title: string; body: string } | null>(null)
  // This projection reads identities/status, not token text. Cache it so native
  // streaming frames do not add another full conversation-tree walk.
  const activity = computed(() => desktopActivity(chat.active))
  const modelHeader = computed(studioModelHeader)
  const documentSelection = ref<{ messageId?: string; attachmentId: string } | null>(null)
  const selectedDocument = computed(() => {
    const selection = documentSelection.value, c = chat.active
    if (!selection || !c) return null
    try {
      if (selection.messageId) return sentDocument(c, selection.messageId, selection.attachmentId)
      const part = staged.get(selection.attachmentId)
      return part ? documentPreview(c, part, 'draft') : null
    } catch { return null } // A removed attachment/branch cannot remain visible.
  })
  let draft = ''
  const toolQuery = ref('')
  const receipts = new Map<string, Promise<unknown>>()
  let mounted!: () => void, closed = false, timer = 0, revision = 0, stopRequested = false
  const initialized = new Promise<void>(resolve => { mounted = resolve })
  const host: NativeContentHost = {
    preview, audioPreview, mounted,
    nativeMarkdown: computed(() => true),
    graphVisible,
    graphArtifact,
    document: computed(() => graphVisible.value ? null : selectedDocument.value?.conversation ?? null),
    documentActions(actions) { documentActions = actions },
    viewport(left, width) {
      if (Math.abs(left - viewport.value.left) > .5 || Math.abs(width - viewport.value.width) > .5) viewport.value = { left, width }
    },
  }
  const audio = createNativeAudio({
    idle,
    async begin() {
      if (staged.size) throw new Error('Remove staged attachments before starting a live transcription')
      const original = conversation(), c = chat.isDraft(original) ? chat.commitDraft(original.model) : original
      await router.push({ name: 'chat', params: { id: c.id } })
    },
    async record(file, text) {
      if (file.size > 100 * 1024 * 1024) throw new Error('Recording exceeds the 100 MiB attachment limit')
      pending.value = true
      try {
        const part = recordedAttachment.value?.type === 'audio' && staged.has(recordedAttachment.value.attachmentId)
          ? recordedAttachment.value : await readAudioPart(file, conversation().id, askedLanguage(conversation().audioLanguage))
        for (const [id, existing] of staged) if (existing.type === 'audio') staged.delete(id)
        staged.set(part.attachmentId, part); recordedAttachment.value = part
      } finally { pending.value = false }
      return execute('send', { text, attachments: [...staged.keys()].map(id => ({ id })) }, crypto.randomUUID())
    },
  })
  const busy = computed(() => pending.value || stream.isStreaming.value || audio.busy.value)
  // Context is sampled on draft/turn/branch changes, not on token body patches.
  watch(() => [busy.value, chat.active?.id, chat.active?.leafId, chat.active?.systemPrompt], () => {
    contextUsed.value = contextTokens(chat.active, draft)
  }, { immediate: true })
  // Stop can arrive during capability/attachment preflight, before the shared
  // orchestrator registers its request controller. Catch that admission edge
  // synchronously so a late-starting request cannot escape the user's Stop.
  watch(stream.isStreaming, active => { if (active && stopRequested) stream.stop() }, { flush: 'sync' })
  const groups = computed(() => {
    const model = chat.active?.model ?? models.currentId, cap = models.caps[model], port = models.portFor(model)
    const values = [{ id: 'artifacts', label: 'Artifacts', key: builtinKey('artifacts'), connectorId: '' }]
    if (graphs.active) values.push({ id: 'graph', label: 'Graph', key: builtinKey('graph'), connectorId: '' })
    for (const label of cap?.mcpServers ?? []) if (port) values.push({ id: label, label, key: serverKey(port, label), connectorId: '' })
    for (const c of connectors.list) if (!c.system && !values.some(v => v.id === c.label)) values.push({ id: c.label, label: c.label, key: connectorKey(c.id), connectorId: c.id })
    return values.map(g => ({ ...g, tools: tools.get(g.key)?.tools ?? [], status: tools.get(g.key)?.status ?? 'idle' }))
  })
  function state() {
    const c = chat.active, id = c?.model ?? models.currentId, caps = models.caps[id]
    const ladder = models.reasoningLadderFor(id), doc = docContext(c)
    const projectHistory = (c: Conversation) => ({ id: c.id, title: c.title.slice(0, 512), model: c.model, updatedAt: c.updatedAt, pinned: !!c.pinned, kind: c.kind ?? 'chat', busy: conversationBusy(c.id), titleState: chat.titleState[c.id] ?? '' })
    const page = libraryRows.value
    return {
      version: VERSION, revision: ++revision,
      activity: { ...activity.value, completedTurn: completedTurn.value },
      nativeTranscript: transcript.value,
      nativeAudioPreview: audioPreview.value ? audioBadge(audioPreview.value) : null,
      nativeGraph: { available: !!graphs.active, visible: !!graphArtifact.value || (graphVisible.value && !!graphs.active && !graphs.folded) },
      nativeArtifacts: artifacts.list,
      markUnsure: settings.markUnsure,
      nativeDocument: graphVisible.value ? null : selectedDocument.value?.badge ?? null,
      conversation: c && !chat.isDraft(c) ? { id: c.id, title: c.title.slice(0, 512), model: c.model, messageCount: c.messages.length } : null,
      historyTotal: chat.conversations.length,
      history: recentRows.value.map(projectHistory),
      library: { ...page, rows: page.rows.map(projectHistory), search: libraryQuery.value.search, sort: libraryQuery.value.sort },
      autoTitle: settings.autoTitle,
      models: models.models.filter(m => takesTurns(m.kind)).map(m => ({ id: m.id, title: m.display ?? m.id, provider: m.cloud?.endpointName ?? 'Local', vendor: m.vendor ?? '', status: m.status, port: m.port, vision: models.visionFor(m.id), audio: models.canTranscribe(m.id), chat: models.canChat(m.id) })),
      selectedModels: c?.compareModels?.length ? c.compareModels : id && id !== 'default' ? [id] : [],
      modelHeader: modelHeader.value,
      audio: { ...audio.state(), attachment: recordedAttachment.value?.type === 'audio' && staged.has(recordedAttachment.value.attachmentId) ? recordedAttachment.value : null },
      busy: busy.value, loading: chat.activeLoading || models.loading, error: error.value || models.error || '',
      viewport: viewport.value, previewing: !!preview.value || !!audioPreview.value,
      unsavedEdits: Object.keys(artifacts.drafts).length > 0,
      settings: { systemPrompt: c?.systemPrompt ?? '', params: c?.params ?? DEFAULT_PARAMS, toolSelection: c ? toolSelection(c) : { mode: 'all' }, connectorIds: c?.connectorIds ?? [], webSearchEnabled: c?.webSearchEnabled ?? true, ocrMode: c?.ocrMode ?? '', audioLanguage: c?.audioLanguage ?? '', maxTokens: settings.maxTokens, summarize: settings.summarize },
      capabilities: { reasoning: models.reasoningStyleFor(id), levels: ladder.levels, reasoningDefault: ladder.opens, reasoningOff: ladder.off, preserveThinking: ladder.preserve, thinkingBudget: models.thinkingBudgetFor(id), webSearch: models.webSearchFor(id), vision: models.visionFor(id), context: models.ctxFor(id), ocrModes: caps?.ocr?.modes ?? [], docParser: caps?.docParser ?? false, taskTags: caps?.taskTags ?? [], pdfRaster: models.pdfRaster },
      tools: groups.value.flatMap(g => {
        const state = { selection: c ? toolSelection(c) : { mode: 'all' as const }, connectorIds: c?.connectorIds ?? [] }
        const group = { label: g.id, connectorId: g.connectorId }, terms = toolQuery.value.trim().toLowerCase().split(/\s+/).filter(Boolean)
        const labelHit = terms.every(t => toolTermHits(t, g.label))
        const visible = labelHit ? g.tools : g.tools.filter(v => terms.every(t => toolTermHits(t, v.name, v.description)))
        if (terms.length && !labelHit && !visible.length) return []
        return [{ ...g, checked: pickerGroupState(state, group), total: g.tools.length,
          selectedCount: g.tools.filter(t => pickerToolChecked(state, group, t.name)).length,
          tools: visible.map(t => ({ name: t.name, description: t.description, selected: pickerToolChecked(state, group, t.name) })) }]
      }),
      composer: { ...composerPresentation(c, [...staged.values()], groups.value), contextUsed: contextUsed.value },
      document: doc ? { id: doc.source.id, name: doc.pdf?.name ?? doc.docx?.name ?? doc.images[0]?.name ?? 'Document' } : null,
    }
  }
  function publish() {
    if (closed) return
    // A throttle, not a trailing debounce: a continuous token stream must not
    // indefinitely postpone the native transcript's next visible update.
    if (timer) return
    timer = window.setTimeout(() => {
      timer = 0
      const value = state()
      // WebKit's structured bridge must receive plain JSON, not Vue proxies
      // (nested proxy dictionaries can lose keys during JSC/Foundation export).
      try {
        for (const frame of presentationFrames(value)) window.webkit?.messageHandlers?.paddockPresentation?.postMessage(frame)
      } catch (e) {
        window.webkit?.messageHandlers?.paddockPresentation?.postMessage({ presentationError: e instanceof Error ? e.message : String(e) })
      }
    }, 32)
  }
  // Stable message IDs allow Swift to retain each incremental parser.
  const unwatch = watch(() => JSON.stringify(state()), publish, { immediate: true })
  watch(() => artifacts.list.find(a => a.id === graphArtifact.value?.id)?.updatedAt, async (updated, previous) => {
    if (closed || updated === undefined || previous === undefined || updated === previous) return
    const selected = graphArtifact.value, c = chat.active?.id
    if (!selected) return
    const content = await artifacts.fetchOne(selected.id)
    if (!closed && chat.active?.id === c && graphArtifact.value?.id === selected.id
      && artifacts.list.find(a => a.id === selected.id)?.updatedAt === updated) {
      graphArtifact.value = { ...selected, body: content.body }
    }
  })
  watch(() => [chat.active?.model, models.caps[chat.active?.model ?? '']], ([id]) => {
    if (typeof id === 'string' && id && !models.caps[id]) void models.capsFor(id)
  }, { immediate: true })
  async function refreshTools() {
    await connectors.refresh()
    tools.ensureBuiltin('artifacts')
    if (graphs.active) tools.ensureBuiltin('graph')
    const id = chat.active?.model ?? models.currentId, port = models.portFor(id)
    if (port) for (const label of models.caps[id]?.mcpServers ?? []) tools.ensureServer(port, label)
    for (const c of connectors.list) tools.ensureConnector(c.id)
  }
  function conversation() {
    const c = chat.active
    if (!c || chat.activeLoading || chat.activeLoadFailed) throw new Error('Wait for the conversation to open')
    return c
  }
  function idle() { if (busy.value) throw new Error('Stop the response before changing this conversation') }
  function applySettings(p: Record<string, unknown>) {
    idle()
    const original = conversation(), c = { ...original }
    let nextLimit = settings.maxTokens, nextSummarize = settings.summarize
    if (p.params !== undefined) validateSamplingPatch(object(p.params))
    if (p.systemPrompt !== undefined) c.systemPrompt = string(p.systemPrompt, 128 * 1024)
    if (p.params !== undefined) {
      const values = object(p.params)
      const out = { ...c.params }
      for (const [key, value] of Object.entries(values)) {
        if (!(key in DEFAULT_PARAMS) && key !== 'thinkingBudget') throw new Error('Unknown sampling parameter')
        if (['thinking', 'preserveThinking'].includes(key)) {
          if (typeof value !== 'boolean') throw new Error('Invalid reasoning switch')
        } else if (key === 'reasoningEffort') string(value, 32)
        else if (key === 'stop') {
          if (!Array.isArray(value) || value.length > 16) throw new Error('Invalid stop sequences')
          value.forEach(v => string(v, 1024))
        } else if (value !== null && (typeof value !== 'number' || !Number.isFinite(value))) throw new Error('Invalid sampling value')
        Object.assign(out, { [key]: value })
      }
      c.params = out as SamplingParams
    }
    if (p.toolSelection !== undefined) {
      const value = object(p.toolSelection)
      if (value.mode === 'all') c.toolSelection = { mode: 'all' }
      else if (value.mode === 'custom' && Array.isArray(value.picks) && value.picks.length <= 256) {
        c.toolSelection = { mode: 'custom', picks: value.picks.map(v => { const pick = object(v); return { label: string(pick.label, 128), ...(pick.tool === undefined ? {} : { tool: string(pick.tool, 256) }) } }) } as ToolSelection
      } else throw new Error('Invalid tool selection')
    }
    if (p.connectorIds !== undefined) {
      if (!Array.isArray(p.connectorIds) || p.connectorIds.length > 128) throw new Error('Invalid connector selection')
      c.connectorIds = p.connectorIds.map(identifier)
      if (c.connectorIds.some(id => !connectors.list.some(v => v.id === id))) throw new Error('A selected connector no longer exists')
    }
    for (const key of ['webSearchEnabled', 'summarize']) if (p[key] !== undefined) {
      if (typeof p[key] !== 'boolean') throw new Error('Invalid switch')
      if (key === 'summarize') nextSummarize = p[key] as boolean
      else c.webSearchEnabled = p[key] as boolean
    }
    if (p.maxTokens !== undefined) {
      if (p.maxTokens !== null && (!Number.isSafeInteger(p.maxTokens) || Number(p.maxTokens) < 1)) throw new Error('Invalid reply limit')
      nextLimit = p.maxTokens as number | null
    }
    if (p.ocrMode !== undefined) c.ocrMode = string(p.ocrMode, 128) || undefined
    if (p.audioLanguage !== undefined) c.audioLanguage = string(p.audioLanguage, 32) || undefined
    Object.assign(original, c)
    settings.maxTokens = nextLimit; settings.summarize = nextSummarize
    chat.persist(original)
  }
  async function execute(kind: string, p: Record<string, unknown>, requestId: string): Promise<unknown> {
    if (!['composerSize', 'draft'].includes(kind)) error.value = ''
    switch (kind) {
      case 'microphoneStart':
        if (staged.size && audio.state().mode !== 'dictate') throw new Error('Send or remove staged attachments before recording')
        await audio.start(); break
      case 'microphoneStop': return { ...await audio.stop(string(p.text ?? '', 128 * 1024)) as object, state: state() }
      case 'microphoneCancel':
        audio.cancel()
        if (recordedAttachment.value?.type === 'audio') staged.delete(recordedAttachment.value.attachmentId)
        recordedAttachment.value = null; break
      case 'microphoneSettings': idle(); await audio.configure(p); break
      case 'microphoneRefresh': await audio.refresh(); break
      case 'transcriptExport': return { export: exportTranscript(conversation(), p) }
      case 'transcriptMarks':
        if (typeof p.enabled !== 'boolean') throw new Error('Invalid transcript mark setting')
        settings.markUnsure = p.enabled; break
      case 'microphoneDevices': await audio.revealDevices(); break
      case 'dictationAck': audio.acknowledge(p); break
      case 'promptList': case 'promptGet': case 'promptSave': case 'promptDelete': return promptCommand(kind, p)
      case 'preferencesGet': {
        if (!models.maxCtx) await models.fetchLimits()
        return { preferences: preferencePresentation() }
      }
      case 'preferencesSave': {
        idle(); pending.value = true
        try { return await saveStudioPreferences(p) } finally { pending.value = false }
      }
      case 'instructionsGet': {
        const c = conversation()
        const instructions = { conversationId: c.id, body: c.systemPrompt, blocks: injected.blocks.value }
        if (new TextEncoder().encode(JSON.stringify(instructions)).length > 200 * 1024) throw new Error("These tool instructions exceed the editor's 200 KiB safety limit.")
        return { instructions }
      }
      case 'instructionsApply': {
        idle()
        const c = conversation(), previous = c.systemPrompt
        if (identifier(p.conversationId) !== c.id || string(p.expected, 128 * 1024) !== previous) throw new Error('The conversation instructions changed. Reopen this editor; your draft is kept.')
        const body = presetText(p.body, 128 * 1024)
        pending.value = true; c.systemPrompt = body
        try { await chat.persistNow(c, true) } catch (e) { c.systemPrompt = previous; throw e }
        finally { pending.value = false }
        break
      }
      case 'historyFilter': libraryQuery.value = historyQuery(p); break
      case 'autoTitle': {
        if (typeof p.enabled !== 'boolean') throw new Error('Invalid automatic-title preference')
        await saveStudioPreferences({ changes: { autoTitle: p.enabled }, expected: { autoTitle: settings.autoTitle } })
        break
      }
      case 'renameChat': case 'pinChat': case 'generateTitle': {
        const id = identifier(p.id)
        if (!chat.conversations.some(c => c.id === id)) throw new Error('Conversation not found')
        if (kind === 'renameChat') {
          const title = string(p.title, 512).trim()
          if (!title || /[\r\n\u0000-\u001f]/.test(title)) throw new Error('Enter a title on one line')
          await chat.rename(id, title)
        } else if (kind === 'pinChat') await chat.togglePin(id)
        else {
          if (busy.value || conversationBusy(id)) throw new Error('Wait for the response before generating a title')
          await chat.generateTitle(id)
        }
        break
      }
      case 'deleteChats': {
        if (!Array.isArray(p.ids) || !p.ids.length || p.ids.length > 500) throw new Error('Choose one to 500 conversations')
        const ids = [...new Set(p.ids.map(identifier))]
        if (ids.some(id => !chat.conversations.some(c => c.id === id))) throw new Error('A selected conversation no longer exists')
        if (ids.some(conversationBusy) || (pending.value && ids.includes(chat.activeId ?? ''))) throw new Error('Stop the response before deleting it')
        const removesActive = ids.includes(chat.activeId ?? '')
        if (removesActive && (staged.size || draft.trim() || Object.keys(artifacts.drafts).length)) throw new Error('Send or discard your draft before deleting this conversation')
        try { await chat.removeMany(ids) } finally {
          // A partial failure still follows the surviving selection. Never
          // resurrect a successfully deleted chat by leaving its route active.
          if (removesActive && !chat.conversations.some(c => ids.includes(c.id) && c.id === chat.activeId)) {
            preview.value = null; documentSelection.value = null
            if (chat.activeId) { await router.push({ name: 'chat', params: { id: chat.activeId } }); await chat.ensureLoaded(chat.activeId) }
            else { await router.push({ name: 'home' }); chat.startDraft(models.currentId || 'default') }
          }
        }
        break
      }
      case 'renderer': {
        if (p.mode !== 'native') throw new Error('Paddock uses native rendering; web chat is not available')
        break
      }
      case 'toolApproval': {
        const c = conversation(), decision = pendingApproval(c, p)
        const m = c.messages.find(m => m.id === p.messageId)!
        const result = await approvalsApi.approve(decision.id, decision.approve, m.model ?? m.run?.model ?? c.model)
        if (!result.ok) throw new Error('The tool request could not be resolved. It may have expired.')
        break
      }
      case 'graphPanel': {
        if (typeof p.open !== 'boolean') throw new Error('Invalid graph visibility')
        graphArtifact.value = null; graphVisible.value = p.open; graphs.folded = !p.open
        if (p.open) documentSelection.value = null
        break
      }
      case 'graphArtifact': {
        const id = identifier(p.id), c = conversation()
        const meta = artifacts.list.find(a => a.id === id && a.kind === 'graph')
        if (!meta) throw new Error('This graph artifact is no longer in the conversation')
        const content = await artifacts.fetchOne(id)
        if (chat.active?.id !== c.id) throw new Error('The conversation changed')
        documentSelection.value = null; graphVisible.value = true
        graphArtifact.value = { id, title: meta.title, body: content.body }
        break
      }
      case 'draft': draft = string(p.text, 128 * 1024); contextUsed.value = contextTokens(chat.active, draft); break
      case 'samplerDefaults': applySettings({ params: samplerDefaults() }); break
      case 'composerSize': {
        if (typeof p.height !== 'number' || !Number.isFinite(p.height) || p.height < 0 || p.height > 2000) throw new Error('Invalid composer inset')
        document.documentElement.style.setProperty('--native-composer-height', `${p.height}px`)
        break
      }
      case 'refresh': {
        await artifacts.refresh(chat.active?.id ?? '')
        tools.invalidate()
        models.invalidateCaps()
        await models.refresh()
        await audio.refresh()
        const id = chat.active?.model ?? models.currentId
        if (id) await models.capsFor(id, true)
        await refreshTools()
        break
      }
      case 'newChat': idle(); preview.value = null; documentSelection.value = null; staged.clear(); await router.push({ name: 'home' }); chat.startDraft(models.currentId || 'default'); break
      case 'open': {
        idle(); const id = identifier(p.id)
        if (!chat.conversations.some(c => c.id === id)) throw new Error('Conversation not found')
        preview.value = null; documentSelection.value = null; staged.clear()
        await router.push({ name: 'chat', params: { id } }); await chat.ensureLoaded(id)
        if (chat.activeLoadFailed) throw new Error('The conversation could not be opened')
        break
      }
      case 'models': {
        idle()
        if (!Array.isArray(p.ids) || p.ids.length < 1 || p.ids.length > 4) throw new Error('Choose one to four models')
        const ids = [...new Set(p.ids.map(v => string(v, 1024)))]
        if (ids.some(id => !models.models.some(m => m.id === id && takesTurns(m.kind) && m.status === 'ok'))) throw new Error('A selected model is not reachable')
        if (!ids.every(id => models.canChat(id)) && !ids.every(id => models.canTranscribe(id))) throw new Error('Compare models must share a text or audio input mode')
        selectStudioModel(ids[0])
        const c = conversation(); c.compareModels = ids.length > 1 ? ids : undefined
        for (const id of ids) await models.capsFor(id)
        chat.persist(c); break
      }
      case 'settings': applySettings(p); await savePreferences(); break
      case 'tools': {
        toolQuery.value = ''
        const id = chat.active?.model ?? models.currentId
        if (id) await models.capsFor(id, true)
        await refreshTools(); break
      }
      case 'toolQuery': toolQuery.value = string(p.query, 256); break
      case 'toolPicker': {
        idle()
        const c = conversation(), kind = string(p.action, 16)
        if (!['all', 'clear', 'group', 'tool'].includes(kind)) throw new Error('Invalid tool action')
        const label = kind === 'all' || kind === 'clear' ? '' : string(p.label, 128)
        const g = groups.value.find(g => g.id === label)
        const action = { kind, label, ...(kind === 'tool' ? { tool: string(p.tool, 256) } : {}) } as PickerAction
        const before = { toolSelection: c.toolSelection, connectorIds: c.connectorIds }
        const next = changeToolPicker({ selection: toolSelection(c), connectorIds: c.connectorIds ?? [] },
          g && { label: g.id, connectorId: g.connectorId, tools: g.status === 'ok' ? g.tools.map(t => t.name) : undefined },
          connectors.list.filter(c => !c.system).map(c => ({ label: c.label, connectorId: c.id })), action)
        c.toolSelection = next.selection; c.connectorIds = next.connectorIds
        try { await chat.persistNow(c, true) } catch (e) { Object.assign(c, before); throw e }
        break
      }
      case 'stage': {
        const a = storedAttachment(p)
        if (staged.size >= 32 && !staged.has(a.id)) throw new Error('Attach at most 32 files per turn')
        const part = staged.get(a.id) ?? await describeAttachment(a)
        staged.set(a.id, part)
        return { attachment: part, state: state() }
      }
      case 'removeAttachment': {
        const id = identifier(p.id)
        if (audioPreview.value?.attachmentId === id) audioPreview.value = null
        staged.delete(id); preview.value = null
        if (!documentSelection.value?.messageId && documentSelection.value?.attachmentId === id) documentSelection.value = null
        break
      }
      case 'preview': {
        graphArtifact.value = null
        graphVisible.value = false
        const id = identifier(p.id), part = staged.get(id), c = conversation()
        if (!part) throw new Error('The attachment is no longer staged')
        if (part.type === 'audio') { audioPreview.value = part; preview.value = null; documentSelection.value = null; break }
        audioPreview.value = null
        if (part.type !== 'image' && !(part.type === 'file' && (isPdfPart(part) || isDocxPart(part)))) throw new Error('This format can be attached, but has no document preview')
        documentPreview(c, part, 'draft') // Validate before changing either presentation.
        documentSelection.value = { attachmentId: id }
        preview.value = { ...c, id: `preview-${id}`, leafId: id, messages: [{ id, role: 'user', content: [part], createdAt: Date.now(), parentId: null }], activeDocId: id }
        break
      }
      case 'openDocument': {
        graphArtifact.value = null
        graphVisible.value = false
        const messageId = identifier(p.messageId), attachmentId = identifier(p.attachmentId)
        sentDocument(conversation(), messageId, attachmentId)
        documentSelection.value = { messageId, attachmentId }; preview.value = null
        break
      }
      case 'documentAction': {
        if (!transcript.value?.available || !selectedDocument.value || !documentActions) throw new Error('Open a document first')
        if (p.action === 'info') documentActions.info()
        else if (p.action === 'download') documentActions.download()
        else throw new Error('Unknown document action')
        break
      }
      case 'closePreview': audioPreview.value = null; preview.value = null; documentSelection.value = null; if (!transcript.value?.available) chat.setDocPane(false); break
      case 'quote': return { text: window.getSelection()?.toString().slice(0, 128 * 1024) ?? '', state: state() }
      case 'stop': stopRequested = true; if (audio.busy.value) await audio.stop(draft); else stream.stop(); break
      case 'messageAction': {
        idle()
        const c = conversation()
        if (Object.keys(artifacts.drafts).length) throw new Error('Save or revert artifact edits first')
        const action = resolveMessageAction(c, p)
        if (action.action === 'branch') {
          pending.value = true
          const previous = { leafId: c.leafId, branchMemory: c.branchMemory ? { ...c.branchMemory } : undefined }
          try {
            if (!chat.selectSibling(action.message.id, action.delta)) throw new Error('The branch could not be selected')
            await chat.persistNow(c, true)
          } catch (e) { Object.assign(c, previous); throw e }
          finally { pending.value = false }
          return { accepted: true, state: state() }
        }
        // Continue uses the original answer's model, as the shared stream
        // does; retry/edit use the current composer model. Never silently
        // substitute a different reachable endpoint for either choice.
        const model = action.action === 'continue' ? action.message.model ?? action.message.run?.model ?? c.model : c.model
        if (!models.models.some(m => m.id === model && m.status === 'ok' && models.canChat(model))) throw new Error('The model for this action is not reachable')
        const presentation = composerPresentation(c, action.parts, groups.value)
        if (action.action === 'edit' && presentation.inputIssue) throw new Error(presentation.inputIssue)
        pending.value = true; stopRequested = false
        let accept!: () => void, reject!: (e: unknown) => void, admitted = false
        const receipt = new Promise<void>((resolve, fail) => { accept = resolve; reject = fail })
        const opts = { accepted: () => { admitted = true; accept() }, beforeRun: () => { if (stopRequested) throw new Error('Stopped before model execution') } }
        const running = action.action === 'edit' ? stream.editAndResend(action.message.id, action.parts, opts)
          : action.action === 'retry' ? stream.regenerate(opts) : stream.continueLast(opts)
        void (async () => {
          let failed = false
          try {
            await running
            if (!admitted) throw new Error('The message action was not accepted')
          } catch (e) {
            failed = !stopRequested; reject(e)
            error.value = e instanceof Error ? e.message : String(e)
            // Preflight can fail before the stream installs its finally block.
            for (const m of c.messages) if (m.streaming) {
              m.streaming = false
              if (stopRequested) m.stopped = true
              else m.error = error.value
            }
          } finally {
            if (admitted) {
              try { await chat.persistNow(c, true) } catch (e) { failed = true; error.value = `Could not save the response: ${String(e)}` }
              const replies = desktopActivity(c).replies
              completedTurn.value = { id: requestId, conversationId: c.id,
                state: stopRequested ? 'stopped' : failed || replies.some(r => ['failed', 'incomplete'].includes(r.state)) ? 'failed' : 'completed' }
            }
            pending.value = false; publish()
          }
        })()
        await receipt
        preview.value = null
        return { accepted: true, requestId, state: state() }
      }
      case 'send': {
        idle()
        const c0 = conversation(), text = string(p.text, 128 * 1024), choices = attachmentChoices(p.attachments ?? [])
        const selected = choices.map(choice => selectedAttachment(staged.get(choice.id), choice))
        const presentation = composerPresentation(c0, selected, groups.value)
        if (presentation.inputIssue) throw new Error(presentation.inputIssue)
        const ids = c0.compareModels?.length ? c0.compareModels : [c0.model]
        if (ids.some(id => !models.models.some(m => m.id === id && m.status === 'ok'))) throw new Error('Select a reachable model before sending')
        const parts: ContentPart[] = text.trim() && !presentation.audioMode ? [{ type: 'text', text }] : []
        for (const part of selected) {
          if (part.type === 'audio') part.language = askedLanguage(c0.audioLanguage)
          parts.push(part)
        }
        if (!parts.length) throw new Error('Write a message or attach a file')
        pending.value = true; stopRequested = false
        try {
          const c = chat.isDraft(c0) ? chat.commitDraft(c0.model) : c0
          for (const part of parts) if (part.type === 'graph') {
            await graphs.ensure(c.id, part.attachmentId, part.name, undefined, c.messages)
            if (graphs.status !== 'ready') throw new Error(graphs.error || 'The graph could not be opened')
          }
          if (stopRequested) throw new Error('Send cancelled; your draft has been kept')
          let accept!: () => void, reject!: (e: unknown) => void
          let admitted = false, failed = false
          const receipt = new Promise<void>((resolve, fail) => { accept = resolve; reject = fail })
          const running = stream.send(parts, { accepted: () => { admitted = true; accept() }, beforeRun: () => { if (stopRequested) throw new Error('Stopped before model execution') } })
          void running.catch(e => { failed = true; reject(e); error.value = e instanceof Error ? e.message : String(e) }).finally(() => {
            const replies = desktopActivity(c).replies
            if (admitted && replies.length) completedTurn.value = { id: requestId, conversationId: c.id,
              state: failed || replies.some(r => r.state === 'failed' || r.state === 'incomplete') ? 'failed'
                : replies.some(r => r.state === 'stopped') ? 'stopped' : 'completed' }
            pending.value = false; publish()
          })
          await receipt
          const recorded = recordedAttachment.value
          if (recorded?.type === 'audio' && choices.some(c => c.id === recorded.attachmentId)) audio.recordingAccepted()
          for (const choice of choices) staged.delete(choice.id)
          audioPreview.value = null
          preview.value = null
          if (!documentSelection.value?.messageId) documentSelection.value = null
          await router.push({ name: 'chat', params: { id: c.id } })
          return { accepted: true, requestId, state: state() }
        } catch (e) { pending.value = false; throw e }
      }
      case 'shutdown':
        audio.cancel()
        stopRequested = true
        await stopAllStreams()
        await Promise.all(chat.conversations.filter(c => c.messages.length).map(c => chat.persistNow(c, true)))
        await savePreferences(); graphs.release(); preview.value = null; documentSelection.value = null; staged.clear(); closed = true; clearTimeout(timer); unwatch(); break
    }
    await nextTick()
    return { state: state() }
  }
  return {
    host, initialized, state, publish,
    async command(value: unknown) {
      const cmd = parseCommand(value)
      if (['newChat', 'open', 'shutdown'].includes(cmd.kind)) { audioPreview.value = null; graphVisible.value = false; graphArtifact.value = null }
      if (closed) throw new Error('The content workspace is closed')
      const previous = receipts.get(cmd.id)
      if (previous) return previous
      const result = execute(cmd.kind, cmd.payload, cmd.id).then(result => {
        // A large state travels once through the framed presentation channel,
        // not again inside a size-limited command acknowledgment.
        if (result && typeof result === 'object' && 'state' in result && new TextEncoder().encode(JSON.stringify(result)).length > 192 * 1024) {
          const { state: _state, ...reply } = result
          return reply
        }
        return result
      }).catch(e => { error.value = e instanceof Error ? e.message : String(e); throw e }).finally(publish)
      // Retransmission can never create a second user turn in this page.
      receipts.set(cmd.id, result)
      if (receipts.size > 128) receipts.delete(receipts.keys().next().value!)
      return result
    },
    theme(dark: boolean) { settings.theme = dark ? 'dark' : 'light'; document.documentElement.setAttribute('data-theme', settings.theme) },
  }
}

declare global {
  interface Window { webkit?: { messageHandlers?: { paddockPresentation?: { postMessage(value: unknown): void } } } }
}
