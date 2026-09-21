import { watch } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useModelsStore } from '@/stores/models'
import { useArtifactsStore } from '@/stores/artifacts'
import { useGraphsStore } from '@/stores/graphs'
import { useChatStream } from '@/composables/useChatStream'
import { activeMessages } from '@/lib/tree'

/** Shared conversation effects, independent of either renderer's view tree. */
export function useConversationEffects() {
  const chat = useChatStore(), models = useModelsStore(), artifacts = useArtifactsStore(), graphs = useGraphsStore()
  const { send, isStreaming } = useChatStream()
  watch(
    [() => artifacts.graphImportFailure, isStreaming],
    ([failure, busy]) => {
      if (!failure || busy || !chat.active) return
      const f = artifacts.consumeGraphImportFailure()
      if (!f) return
      // The report names its artifact and goes only to the lane whose model
      // wrote it - the first live compare run sent it to both lanes, and the
      // healthy lane started redoing a working graph.
      const meta = artifacts.list.find((a) => a.id === f.artifactId)
      void send(
        [
          {
            type: 'text',
            text:
              `[automatic import report] Your graph artifact ${f.artifactId}` +
              `${meta?.title ? ` ("${meta.title}")` : ''} failed to import - ` +
              'the app runs the script when it renders, and it did not execute ' +
              `cleanly. ${f.summary}\n` +
              `Fix ${f.artifactId} IN PLACE with artifact_update or ` +
              'artifact_rewrite, following every rule in the graph ' +
              'instructions. Do NOT create a new artifact - a fresh ' +
              'artifact_create is the wrong answer. Do not describe the fix - ' +
              'apply it.',
          },
        ],
        { lane: meta?.model || undefined, auto: true },
      )
    },
    { immediate: true },
  )
  watch(
    () => [chat.activeId, isStreaming.value] as const,
    ([id, streaming]) => {
      if (!streaming) void artifacts.refresh(id ?? '')
    },
    { immediate: true },
  )
  watch(
    () => models.currentId,
    (id) => {
      const c = chat.active
      if (!id || !c || !chat.isDraft(c) || c.messages.length) return
      // Compare arms the lane set explicitly; the seat has no business in it.
      if (c.compareModels?.length) return
      c.model = id
    },
  )
  watch(
    // Messages load asynchronously after the id is set, so keying on the id
    // alone scanned an empty list and never looked again - a reopened graph
    // conversation showed nothing (the maintainer, first live session). The length makes
    // the scan re-run when the doc actually arrives; ensure() is idempotent so
    // the extra fires are free.
    () => [chat.active?.id, chat.active?.messages.length, chat.active?.leafId] as const,
    ([id]) => {
      const conv = chat.active
      if (!id || !conv) {
        graphs.release()
        return
      }
      let g: { attachmentId: string; name: string } | undefined
      // The branch on screen: a graph on a branch the user switched away from
      // must not keep driving the pane.
      for (const m of activeMessages(conv)) {
        for (const part of m.content) if (part.type === 'graph') g = part
      }
      if (g) void graphs.ensure(id, g.attachmentId, g.name, undefined, activeMessages(conv))
      // Only a different conversation's session is stale - "no graph part in
      // sight" also happens while this conversation's messages are still
      // loading, and releasing then would kill a session we are about to want.
      else if (graphs.conversationId && graphs.conversationId !== id) graphs.release()
    },
    { immediate: true },
  )
}
