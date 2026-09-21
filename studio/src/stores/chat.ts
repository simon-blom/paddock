import { defineStore } from 'pinia'
import { uuid } from '@/lib/uuid'
import { computed, ref, toRaw } from 'vue'
import type { AudioPart, Conversation, Message, SamplingParams } from '@/types/chat'
import { DEFAULT_PARAMS, messageText } from '@/types/chat'
import { activeMessages, deleteSubtree, migrate, stepSibling, tipId } from '@/lib/tree'
import { taskLabel } from '@/lib/tasks'
import { store, stubFromSummary } from '@/lib/api'
import { titleGenerator, titleTranscript } from '@/lib/chat-title'
import { useModelsStore } from './models'
import { useSettingsStore } from './settings'
import { isConversationRunning } from '@/lib/chat-activity'

function uid(): string {
  return uuid()
}

export const useChatStore = defineStore('chat', () => {
  // Summaries hydrate the list; full messages load lazily on open.
  const conversations = ref<Conversation[]>([])
  const activeId = ref<string | null>(null)
  const loaded = ref(false)
  const loadedIds = ref<Set<string>>(new Set())
  // Documents that were asked for and did not arrive. Kept apart from "not
  // loaded yet" so the view can offer a retry instead of an endless spinner.
  const failedIds = ref<Set<string>>(new Set())
  // One fetch per document however many callers ask at once - hydrate and the
  // route sync both open the resumed chat.
  const inflight = new Map<string, Promise<void>>()
  const metadataEdits = new Map<string, Promise<void>>()
  const deleting = new Set<string>()
  const titleState = ref<Record<string, string>>({})
  const titleJobs = new Map<string, symbol>()

  // The start page's conversation-in-waiting. It is deliberately not in
  // `conversations` and not on the server: the sidebar lists committed chats
  // only, so `/` shows no row and nothing selected. It exists so the composer
  // has something real to write to - its model/system-prompt/thinking toggle
  // all land here - and gets committed by the first send. Abandon the start
  // page and it's simply dropped.
  const draft = ref<Conversation | null>(null)

  /** The conversation in view: the selected one, else the start-page draft. */
  const active = computed(
    () => conversations.value.find((c) => c.id === activeId.value) ?? draft.value,
  )

  function isDraft(c: Conversation): boolean {
    return draft.value?.id === c.id
  }

  /** The chat in view is a committed one whose document has not arrived: the
   *  list row is only a stub with no messages, and drawing it would show an
   *  empty conversation that is not. The list used to take long enough on a
   *  cold disk that the document was cached by the time anyone opened it;
   *  with the list answered from an index the document is now the slow read,
   *  so this state is real and visible for seconds. */
  const activeLoading = computed(() => {
    const c = active.value
    return !!c && !isDraft(c) && !loadedIds.value.has(c.id) && !failedIds.value.has(c.id)
  })
  const activeLoadFailed = computed(() => {
    const c = active.value
    return !!c && !isDraft(c) && failedIds.value.has(c.id)
  })

  // Both the shell (for the nav's chat count) and ChatView hydrate; share the
  // in-flight fetch so concurrent callers don't both list + both set activeId.
  let hydrating: Promise<void> | null = null

  async function hydrate(): Promise<void> {
    if (loaded.value) return
    if (hydrating) return hydrating
    hydrating = (async () => {
      try {
        const summaries = await store.listConversations()
        conversations.value = summaries.map(stubFromSummary)
      } catch (e) {
        console.error('failed to load conversations', e)
        conversations.value = []
      }
      loaded.value = true
      activeId.value = lastOpenId()
      if (activeId.value) await ensureLoaded(activeId.value)
    })()
    try {
      await hydrating
    } finally {
      hydrating = null
    }
  }

  /** The chat to resume when no id is given: the last one opened, else the
   *  newest, else null when there are none.
   *
   *  Reads localStorage rather than `activeId` deliberately - the start page
   *  clears activeId (nothing is selected there), so activeId is null exactly
   *  when someone arrives from `/`. */
  function lastOpenId(): string | null {
    const last = localStorage.getItem('pk_last_conversation')
    if (last && conversations.value.some((c) => c.id === last)) return last
    return conversations.value[0]?.id ?? null
  }

  /** Fetch a conversation's full document (messages) the first time it opens. */
  async function ensureLoaded(id: string): Promise<void> {
    if (loadedIds.value.has(id)) return
    const pending = inflight.get(id)
    if (pending) return pending
    const run = fetchDocument(id).finally(() => inflight.delete(id))
    inflight.set(id, run)
    return run
  }

  async function fetchDocument(id: string): Promise<void> {
    failedIds.value.delete(id)
    try {
      const full = await store.getConversation(id)
      // Pre-popover chats carry the old hard-coded sampling the user never
      // chose (exactly 0.7/0.8/20). Those become "untouched" so nothing is
      // sent - the same values the user did set survive any other combination.
      const p = full.params
      if (p && p.temperature === 0.7 && p.topP === 0.8 && p.topK === 20) {
        p.temperature = null
        p.topP = null
        p.topK = null
      }
      // The penalty fields predate the popover (no UI ever wrote them, always
      // 0): 0 becomes "untouched" instead of claiming an explicit choice.
      if (p) {
        if (p.frequencyPenalty === 0) p.frequencyPenalty = null
        if (p.presencePenalty === 0) p.presencePenalty = null
      }
      // Give the thread its tree links. Every conversation written before
      // branching is a straight line, so this is exact rather than a guess -
      // and it is deliberately not saved back here: the inference is
      // deterministic, so re-deriving it on each open costs nothing and beats
      // rewriting every stored conversation the first time someone opens it.
      // The links persist the moment the chat is next written for a real
      // reason.
      migrate(full)
      const i = conversations.value.findIndex((c) => c.id === id)
      if (i >= 0) conversations.value.splice(i, 1, full)
      loadedIds.value.add(id)
    } catch (e) {
      console.error('failed to load conversation', id, e)
      failedIds.value.add(id)
    }
  }

  /** Generic settings/model edits also apply to in-memory drafts. Keep this
   * separate from acknowledged history metadata operations: a draft has no
   * stored document to hydrate yet. */
  async function edit(id: string, apply: (c: Conversation) => void): Promise<void> {
    const now = conversations.value.find(x => x.id === id) ?? (draft.value?.id === id ? draft.value : null)
    if (!now) return
    apply(now)
    if (isDraft(now)) return
    await ensureLoaded(id)
    const live = conversations.value.find(x => x.id === id)
    if (!live || !loadedIds.value.has(id)) return
    if (toRaw(live) !== toRaw(now)) apply(live)
    await persistNow(live)
  }

  /** Hydrate before editing metadata; never save a summary stub over the full
   * document. Serialize metadata edits and roll back only their fields on a
   * failed durable save, leaving newly streamed message content untouched. */
  function editMetadata(id: string, apply: (c: Conversation) => void): Promise<void> {
    const run = (metadataEdits.get(id) ?? Promise.resolve()).catch(() => {}).then(async () => {
      if (deleting.has(id)) throw new Error('This conversation is being deleted')
      await ensureLoaded(id)
      const live = conversations.value.find(x => x.id === id)
      if (!live || !loadedIds.value.has(id) || deleting.has(id)) throw new Error('The conversation could not be opened')
      const before = { title: live.title, pinned: live.pinned, titleSource: live.titleSource, titleModel: live.titleModel, titleCostUsd: live.titleCostUsd }
      apply(live)
      try { await persistNow(live, true) } catch (e) { Object.assign(live, before); throw e }
    })
    metadataEdits.set(id, run)
    void run.catch(() => {}).finally(() => { if (metadataEdits.get(id) === run) metadataEdits.delete(id) })
    return run
  }

  function makeConversation(model: string, params?: Partial<SamplingParams>): Conversation {
    const now = Date.now()
    return {
      id: uid(),
      title: 'New chat',
      messages: [],
      model,
      systemPrompt: '',
      params: { ...DEFAULT_PARAMS, ...params },
      createdAt: now,
      updatedAt: now,
    }
  }

  /** Put a conversation into the list, select it, and save it - the moment it
   *  becomes real (shows in the sidebar, exists on the server). */
  function commit(c: Conversation): Conversation {
    draft.value = null
    conversations.value.unshift(c)
    activeId.value = c.id
    loadedIds.value.add(c.id)
    localStorage.setItem('pk_last_conversation', c.id)
    void save(c)
    return c
  }

  function newConversation(model: string, params?: Partial<SamplingParams>): Conversation {
    return commit(makeConversation(model, params))
  }

  /** Open the start page: an uncommitted conversation with nothing selected. */
  function startDraft(model: string): Conversation {
    draft.value = makeConversation(model)
    activeId.value = null // no sidebar row is current - there's no chat yet
    return draft.value
  }

  /** First send from the start page: the draft becomes a real conversation,
   *  keeping whatever the composer already put on it (model, system prompt,
   *  thinking). */
  function commitDraft(model: string): Conversation {
    return commit(draft.value ?? makeConversation(model))
  }

  function select(id: string): void {
    draft.value = null // leaving the start page discards its unsent draft
    activeId.value = id
    localStorage.setItem('pk_last_conversation', id)
    void ensureLoaded(id)
  }

  /** Bump updatedAt, re-sort to top, persist (debounced). */
  function touch(c: Conversation): void {
    c.updatedAt = Date.now()
    const i = conversations.value.findIndex((x) => x.id === c.id)
    if (i > 0) {
      conversations.value.splice(i, 1)
      conversations.value.unshift(c)
    }
    persist(c)
  }

  const timers = new Map<string, number>()
  function persist(c: Conversation): void {
    // An uncommitted draft stays in memory. The composer's toggles persist on
    // every change, which would otherwise PUT the start page's chat and leave
    // an orphan on the server that the sidebar never shows.
    if (isDraft(c)) return
    const prev = timers.get(c.id)
    if (prev) clearTimeout(prev)
    timers.set(
      c.id,
      window.setTimeout(() => {
        timers.delete(c.id)
        void save(c)
      }, 400),
    )
  }

  /** Save now, cancelling any pending debounced save. The promise settles
   *  when the write has landed - fire-and-forget callers can ignore it. */
  function persistNow(c: Conversation, strict = false): Promise<void> {
    if (isDraft(c)) return Promise.resolve() // see persist(): drafts are in-memory until sent
    const prev = timers.get(c.id)
    if (prev) {
      clearTimeout(prev)
      timers.delete(c.id)
    }
    return save(c, strict)
  }

  const writes = new Map<string, Promise<void>>()
  function save(c: Conversation, strict = false): Promise<void> {
    // Serialize whole-document replacements per conversation. In particular,
    // committing a draft must not finish its empty save after the first turn's
    // durable receipt. Independent chats still save concurrently.
    const write = (writes.get(c.id) ?? Promise.resolve()).then(() => writeDocument(c, strict))
    const settled = write.catch(() => {})
    writes.set(c.id, settled)
    void settled.then(() => { if (writes.get(c.id) === settled) writes.delete(c.id) })
    return write
  }

  async function writeDocument(c: Conversation, strict: boolean): Promise<void> {
    // The server REPLACES the stored document with whatever is sent. A list
    // stub has no messages, so saving one wipes the conversation's history -
    // and roughly thirty call sites persist whatever conversation they hold.
    // So the choke point refuses two things: a document that has not loaded,
    // and an object that is no longer the live one (a caller still holding
    // the stub after the load replaced it would otherwise pass an id check and
    // write the stub back). A pending save for a chat deleted meanwhile lands
    // here too, and must not bring it back.
    if (!isDraft(c)) {
      const live = conversations.value.find((x) => x.id === c.id)
      if (deleting.has(c.id) || !live || !loadedIds.value.has(c.id) || toRaw(live) !== toRaw(c)) {
        if (strict) throw new Error('The conversation changed before it could be saved')
        console.warn('not saving conversation', c.id, '- its document is not the loaded one')
        return
      }
    }
    try {
      await store.putConversation(c)
    } catch (e) {
      if (strict) throw e
      console.error('failed to save conversation', c.id, e)
    }
  }

  // Rename is metadata, not activity: don't bump updatedAt or re-sort (renaming
  // used to yank the chat to the top of the list - jarring). Persist in place.
  function rename(id: string, title: string): Promise<void> {
    titleGenerator.cancel(id)
    delete titleState.value[id]
    const t = title.trim() || 'Untitled'
    return editMetadata(id, (c) => {
      c.title = t
      c.titleSource = 'manual'
      c.titleModel = undefined
      c.titleCostUsd = undefined
    })
  }

  /** Re-point activeId after the current chat is gone; keeps URL/localStorage in
   *  sync. Returns the new active id (or null when the list is now empty). */
  function repointActive(): string | null {
    activeId.value = conversations.value[0]?.id ?? null
    if (activeId.value) {
      localStorage.setItem('pk_last_conversation', activeId.value)
      void ensureLoaded(activeId.value)
    } else {
      localStorage.removeItem('pk_last_conversation')
    }
    return activeId.value
  }

  async function remove(id: string): Promise<void> {
    const c = conversations.value.find(x => x.id === id)
    if (!c) return
    if (deleting.has(id)) throw new Error('This conversation is already being deleted')
    if (isConversationRunning(id)) throw new Error('Stop the response before deleting this conversation')
    titleGenerator.cancel(id)
    deleting.add(id)
    const timer = timers.get(id)
    if (timer) clearTimeout(timer)
    timers.delete(id)
    try {
      // Drain a PUT already on the wire before DELETE, preventing resurrection.
      await metadataEdits.get(id)?.catch(() => {})
      await writes.get(id)
      await store.deleteConversation(id)
      conversations.value = conversations.value.filter(x => x.id !== id)
      loadedIds.value.delete(id)
      failedIds.value.delete(id)
      delete titleState.value[id]
      if (activeId.value === id) repointActive()
    } finally { deleting.delete(id) }
  }

  /** Delete several chats at once (multi-select). Re-points the active chat once
   *  if it was among them. */
  async function removeMany(ids: string[]): Promise<void> {
    const failures: string[] = []
    for (const id of new Set(ids)) {
      try { await remove(id) } catch { failures.push(id) }
    }
    if (failures.length) throw new Error(`${failures.length} conversation(s) could not be deleted and remain in your library`)
  }

  // Pinning is metadata too - persist without a reorder/updatedAt bump; the
  // sidebar floats pinned chats to the top via its sort.
  function togglePin(id: string): Promise<void> {
    const c = conversations.value.find((x) => x.id === id)
    if (!c) return Promise.resolve()
    // decided once, from what the user saw - re-reading it off the loaded
    // document would flip it back
    const pinned = !c.pinned
    return editMetadata(id, (x) => {
      x.pinned = pinned
    })
  }

  /** Set the title from the first user message, once, if still default.
   *
   *  An AUDIO turn has no words of its own, so the title comes from what was
   *  SAID - which only exists once the transcript lands, hence this runs
   *  again when a transcription finishes. Until then it declines rather than
   *  locking in a placeholder. The file name is the fallback, and it is only
   *  a fallback: a microphone recording has no name at all, which is exactly
   *  why the transcript leads. */
  function maybeTitle(c: Conversation): void {
    if (c.title !== 'New chat' || c.titleSource === 'manual') return
    // The branch on screen: if the opening question was edited before a
    // title existed, the title should come from the question being asked, not
    // from the one that was replaced.
    const path = activeMessages(c)
    const first = path.find((m) => m.role === 'user')
    if (!first) return
    let raw = messageText(first).replace(/\s+/g, ' ').trim()
    if (!raw) {
      const clip = first.content.find((p): p is AudioPart => p.type === 'audio')
      if (!clip) {
        const attachment = first.content.find(p => 'name' in p && p.name)
        if (attachment && 'name' in attachment) raw = attachment.name
        if (!raw) return
      } else {
        // Wait for all lanes, then prefer the first successful transcript.
        const answers: Message[] = []
        for (let i = path.indexOf(first) + 1; i < path.length; i++) {
          if (path[i].role !== 'assistant') break
          answers.push(path[i])
        }
        if (!answers.length || answers.some((m) => m.streaming)) return
        raw = answers.map(m => messageText(m).replace(/\s+/g, ' ').trim()).find(Boolean) ?? clip.name
        if (!raw) return
      }
    }
    // A chat opened with a task action is titled by the action's plain name -
    // the sidebar must not be the one place a markup token leaks into view.
    const t = (taskLabel(raw) ?? raw).slice(0, 48)
    if (t) {
      c.title = t
      c.titleSource = 'fallback'
      persistNow(c)
    }
  }

  /** Only new fallback labels are automatic. Legacy and manually named chats
   * require the explicit Generate title action. Names never mutate context. */
  async function generateTitle(id: string, automatic = false): Promise<void> {
    if (automatic && !useSettingsStore().autoTitle) return
    if (!titleGenerator.available) {
      if (automatic) return
      throw new Error('Wait for active responses before generating a title')
    }
    await ensureLoaded(id)
    const c = conversations.value.find(x => x.id === id)
    if (!c || deleting.has(id) || !loadedIds.value.has(id)) return
    if (automatic && (c.titleSource !== 'fallback' || titleState.value[id])) return
    if (isConversationRunning(id)) throw new Error('Wait for the response before generating a title')
    const models = useModelsStore(), target = models.models.find(m => m.id === c.model && m.status === 'ok')
    if (!target || !models.canChat(c.model)) {
      if (automatic) return
      throw new Error("Start this conversation's text model to generate its title")
    }
    const endpoint = models.responsesUrl(c.model)
    if (!endpoint) throw new Error("The conversation's model is not reachable")
    const input = titleTranscript(c)
    if (!input) return
    const previous = { title: c.title, source: c.titleSource, leaf: c.leafId }
    const job = Symbol(id)
    titleJobs.set(id, job)
    const ladder = models.reasoningLadderFor(c.model), style = models.reasoningStyleFor(c.model)
    titleState.value[id] = 'generating'
    try {
      const result = await titleGenerator.generate(id, input, { model: c.model, endpoint,
        ...(style === 'effort' ? { reasoning: { effort: ladder.off ? 'none' : ladder.levels[0] || 'low' } } : {}),
        ...(style === 'toggle' && !target.cloud ? { chat_template_kwargs: { enable_thinking: false } } : {}),
      })
      await editMetadata(id, live => {
        if (toRaw(live) !== toRaw(c) || live.title !== previous.title || live.titleSource !== previous.source || live.leafId !== previous.leaf) {
          throw new Error('The conversation changed; the generated title was not applied')
        }
        live.title = result.title; live.titleSource = 'generated'; live.titleModel = c.model; live.titleCostUsd = result.cost
      })
      if (titleJobs.get(id) === job) delete titleState.value[id]
    } catch (e) {
      if (titleJobs.get(id) !== job) return
      if (e instanceof DOMException && e.name === 'AbortError') { delete titleState.value[id]; return }
      titleState.value[id] = e instanceof Error ? e.message : 'Title generation failed'
      if (!automatic) throw e
    } finally { if (titleJobs.get(id) === job) titleJobs.delete(id) }
  }

  /** Add a turn to the tree; returns the REACTIVE element the array now holds
   *  (so a streaming caller mutates the proxy, not the raw object).
   *
   *  `parentId` is what makes a turn an ALTERNATIVE rather than a
   *  continuation: omit it and the turn extends the branch on screen; pass the
   *  parent of an existing turn and the new one becomes its sibling, which is
   *  all a regenerate or an edited question actually is. Pass `null` for a
   *  second root. */
  function addMessage(c: Conversation, m: Message, parentId?: string | null): Message {
    m.parentId = parentId !== undefined ? parentId : (tipId(c) ?? null)
    c.messages.push(m)
    const added = c.messages[c.messages.length - 1]
    // The cursor follows what was just written - except inside a compare
    // fan-out, where the lanes are one step and the anchor (the first lane to
    // land) is what later turns hang from. Without this the cursor would walk
    // to whichever lane was created last and the group's own children would
    // hang off the wrong message.
    if (m.group) {
      const anchor = c.messages.find((x) => x.group === m.group)
      c.leafId = anchor ? anchor.id : added.id
    } else {
      c.leafId = added.id
    }
    touch(c)
    return added
  }

  /** Show the sibling `delta` steps away from the one holding `id` - the
   *  `< 2/3 >` control under a branched turn. Returns false when there is
   *  nowhere to go, so the caller can disable rather than silently no-op. */
  function selectSibling(id: string, delta: number): boolean {
    const c = active.value
    if (!c) return false
    const moved = stepSibling(c, id, delta)
    if (moved) persist(c)
    return moved
  }

  /** Drop a turn and everything that followed from it. The whole step goes
   *  (half a compare block is not a thing), and the cursor lands on a
   *  surviving branch rather than nowhere. */
  function removeMessage(id: string): void {
    const c = active.value
    if (!c) return
    if (!deleteSubtree(c, id).length) return
    touch(c)
  }

  /** Put a document on screen: select it AND open the pane. The two ways in -
   *  a file chip in the thread and a tab in the pane itself - both land here,
   *  so "selection = target" (lib/docrun.ts) holds however you arrived. */
  function showDocument(sourceId: string): void {
    const c = active.value
    if (!c) return
    c.activeDocId = sourceId
    c.docPaneOpen = true
    persist(c)
  }

  /** Fold the document pane away, or bring it back. Writes the flag even when
   *  it matches what was implied, which is the point: from here on the
   *  conversation has an opinion and the model-derived default stops applying. */
  function setDocPane(open: boolean): void {
    const c = active.value
    if (!c) return
    c.docPaneOpen = open
    persist(c)
  }

  function setArtifactsPane(open: boolean): void {
    const c = active.value
    if (!c) return
    c.artifactsPaneOpen = open
    persist(c)
  }

  return {
    conversations,
    activeId,
    active,
    isDraft,
    loaded,
    activeLoading,
    activeLoadFailed,
    edit,
    showDocument,
    setDocPane,
    setArtifactsPane,
    hydrate,
    lastOpenId,
    ensureLoaded,
    newConversation,
    startDraft,
    commitDraft,
    select,
    touch,
    persist,
    persistNow,
    rename,
    remove,
    removeMany,
    togglePin,
    generateTitle,
    titleState,
    maybeTitle,
    addMessage,
    selectSibling,
    removeMessage,
  }
})
