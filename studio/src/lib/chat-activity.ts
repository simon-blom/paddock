import { shallowReactive } from 'vue'

/** Process-local live controllers, not persisted message.streaming flags.
 * A crash can leave a saved flag behind; it must not make that chat undeletable. */
export const activeChatControllers = shallowReactive(new Map<string, Set<AbortController>>())
export function isConversationRunning(id: string | null | undefined): boolean {
  return !!id && (activeChatControllers.get(id)?.size ?? 0) > 0
}
